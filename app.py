#!/usr/bin/env python3
"""
app.py — Piano Coach  · Auth + Model Server
=============================================
Replaces serve_model.py.  Handles:
  • User registration / login  (JWT tokens)
  • User level management      (beginner / intermediate)
  • Real-time note inference    (POST /api/predict)

SETUP:
    pip install flask flask-cors flask-jwt-extended flask-bcrypt flask-sqlalchemy torch librosa numpy

USAGE:
    python app.py --model "E:\\TWOFYP\\checkpoints\\best_model.pt"

    Then open login.html in your browser.

ENDPOINTS:
    POST /api/register          { username, email, password }
    POST /api/login             { email, password }  → { token, user }
    GET  /api/me                (JWT required)       → user info
    PUT  /api/me/level          (JWT required)       { level: "beginner"|"intermediate" }
    POST /api/sessions         (JWT required)       { song_key, song_name, correct, wrong, accuracy, timing_avg_ms, duration_secs }
    GET  /api/analytics        (JWT required)       → sessions_by_day, streak, totals, song_breakdown, recent
    GET  /api/status            public               → server + model info
    POST /api/predict           (JWT required)       { audio, sr, yin_midi } → predictions
"""

import argparse, sys, os
from pathlib import Path
from datetime import timedelta

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)

import numpy as np
import torch
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from phase3_model import PianoTranscriptionCNN
try:
    from phase3_nsynth_model import NSynthPitchCNN
    NSYNTH_AVAILABLE = True
except ImportError:
    NSYNTH_AVAILABLE = False

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── STATIC FILE SERVING ──────────────────────────────────────────────────────
# Serve all HTML/JS/JSON/PNG files from the same directory as app.py.
# This means you only need ONE ngrok tunnel (port 5000) for everything.

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'login.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

# ngrok free tier adds a browser warning page — this header bypasses it
@app.after_request
def add_ngrok_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

BASE_DIR = Path(__file__).parent

app.config["SECRET_KEY"]             = "piano-coach-secret-change-in-prod"
app.config["JWT_SECRET_KEY"]         = "piano-coach-jwt-secret-change-in-prod"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{BASE_DIR / 'piano_coach.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db      = SQLAlchemy(app)
bcrypt  = Bcrypt(app)
jwt     = JWTManager(app)

# ─── MODELS ───────────────────────────────────────────────────────────────────

VALID_LEVELS = {"beginner", "intermediate", "expert"}

class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer,     primary_key=True)
    username      = db.Column(db.String(80),  unique=True,  nullable=False)
    email         = db.Column(db.String(120), unique=True,  nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    level         = db.Column(db.String(20),  default="beginner", nullable=False)
    created_at    = db.Column(db.DateTime,    server_default=db.func.now())

    def to_dict(self):
        return {
            "id":         self.id,
            "username":   self.username,
            "email":      self.email,
            "level":      self.level,
            "created_at": str(self.created_at),
        }

class Session(db.Model):
    __tablename__ = "sessions"
    id             = db.Column(db.Integer,   primary_key=True)
    user_id        = db.Column(db.Integer,   db.ForeignKey("users.id"), nullable=False, index=True)
    song_key       = db.Column(db.String(40),  nullable=False)
    song_name      = db.Column(db.String(120), nullable=False)
    correct        = db.Column(db.Integer,   default=0)
    wrong          = db.Column(db.Integer,   default=0)
    accuracy       = db.Column(db.Float,     default=0.0)   # 0–100
    timing_avg_ms  = db.Column(db.Float,     nullable=True)  # None if rhythm off
    duration_secs  = db.Column(db.Integer,   default=0)
    played_at      = db.Column(db.DateTime,  server_default=db.func.now(), index=True)

    def to_dict(self):
        return {
            "id":           self.id,
            "song_key":     self.song_key,
            "song_name":    self.song_name,
            "correct":      self.correct,
            "wrong":        self.wrong,
            "accuracy":     round(self.accuracy, 1),
            "timing_avg_ms":self.timing_avg_ms,
            "duration_secs":self.duration_secs,
            "played_at":    self.played_at.strftime("%Y-%m-%d %H:%M") if self.played_at else None,
            "date":         self.played_at.strftime("%Y-%m-%d")        if self.played_at else None,
        }


class MidiReference(db.Model):
    __tablename__ = "midi_references"
    id          = db.Column(db.Integer,    primary_key=True)
    user_id     = db.Column(db.Integer,    db.ForeignKey("users.id"), nullable=False, index=True)
    title       = db.Column(db.String(200), nullable=False)
    composer    = db.Column(db.String(200), default="")
    notes_json  = db.Column(db.Text,       nullable=False)   # JSON [{midi, timeSec}, ...]
    note_count  = db.Column(db.Integer,    default=0)
    uploaded_at = db.Column(db.DateTime,   server_default=db.func.now())


# ─── CNN / INFERENCE GLOBALS ──────────────────────────────────────────────────

CNN_MODEL    = None
DEVICE       = None
MODEL_INFO   = {}
BEST_THRESHOLD = 0.35   # default; overridden by best_threshold.pt when model loads

# ── NSynth note classifier (beginner mode) ────────────────────────────────────
NSYNTH_MODEL      = None
NSYNTH_MODEL_INFO = {}

# ── Temporal smoothing — rolling average of last N predictions ────────────────
# Averaging the last 3 frames suppresses one-off false triggers while
# keeping genuine sustained notes stable.
SMOOTH_WINDOW = 3
_prob_history = {
    "intermediate": [],   # CNN chord probs for intermediate mode
    "expert":       [],   # CNN probs for expert/fallback mode
}   # kept separate so a mode-switch never bleeds stale frames across modes

SAMPLE_RATE  = 16000
HOP_LENGTH   = 512
FFT_SIZE     = 2048
N_MELS       = 64
FMIN         = 27.5
FMAX         = 4200.0
N_FRAMES     = 32      # from 11 — must match build_dataset.py
MIDI_MIN     = 21
N_NOTES      = 88
NOTE_NAMES   = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Audio Preprocessing
def audio_to_mel_window(audio: np.ndarray, sr: int) -> np.ndarray:
    # Resample to model's expected sample rate
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Amplitude normalisation
    # Prevents volume differences from killing predictions.
    # A quiet recording and a loud recording now look the same to the model.
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Ensure audio is long enough for at least one window
    min_len = FFT_SIZE + HOP_LENGTH * N_FRAMES
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_fft=FFT_SIZE, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    # Convert to log scale (dB) and normalise to [0, 1]
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    T      = mel_db.shape[1]
    centre = T // 2
    half   = N_FRAMES // 2
    start  = max(0, centre - half)
    end    = min(T, start + N_FRAMES)
    start  = max(0, end - N_FRAMES)
    window = mel_db[:, start:end]
    return window[np.newaxis, np.newaxis, :, :].astype(np.float32)
    # Output shape: (1, 1, N_MELS, N_FRAMES) — batch of 1, single channel


# Inference logic for all levels. Takes raw audio + YIN MIDI as input, returns fused MIDI + CNN probabilities + metadata.
def run_inference(audio: np.ndarray, sr: int, yin_midi: int, level: str = 'beginner'):
    global _prob_history

    window = audio_to_mel_window(audio, sr)
    tensor = torch.from_numpy(window).to(DEVICE)

    # BEGINNER, YIN + NSynth classifier
    if level == 'beginner' and NSYNTH_MODEL is not None:
        with torch.no_grad():
            logits = NSYNTH_MODEL(tensor)
            probs  = torch.softmax(logits, dim=1)
        probs_np    = probs.cpu().numpy()[0]
        nsynth_idx  = int(np.argmax(probs_np))
        nsynth_midi = MIDI_MIN + nsynth_idx
        nsynth_conf = float(probs_np[nsynth_idx])

        # YIN-first fusion: trust YIN when it agrees with NSynth
        if yin_midi > 0 and abs(yin_midi - nsynth_midi) <= 1:
            fused_midi = yin_midi
            fused_conf = min(0.99, nsynth_conf + 0.05)
        # If NSynth is confident, trust it even if YIN is silent or disagrees, we don't want a high-confidence NSynth prediction to be silently overridden by a random YIN estimate of 0.
        elif nsynth_conf > 0.6:
            fused_midi = nsynth_midi
            fused_conf = nsynth_conf
        # If NSynth is uncertain but YIN has a pitch, trust YIN, better to have a somewhat unreliable MIDI estimate than no estimate at all.
        elif yin_midi > 0:
            fused_midi = yin_midi
            fused_conf = 0.65
        # If both are unreliable, fallback to NSynth's best guess.
        else:
            fused_midi = nsynth_midi
            fused_conf = nsynth_conf

        return {
            "midi":          fused_midi,
            "note":          NOTE_NAMES[fused_midi % 12],
            "octave":        (fused_midi // 12) - 1,
            "confidence":    round(fused_conf, 4),
            "probabilities": probs_np.tolist(),
            "model_used":    "nsynth",
        }

    # INTERMEDIATE, NSynth primary note + CNN chord probs
    elif level == 'intermediate' and NSYNTH_MODEL is not None and CNN_MODEL is not None:
        with torch.no_grad():
            nsynth_logits = NSYNTH_MODEL(tensor)
            nsynth_probs  = torch.softmax(nsynth_logits, dim=1)
            cnn_logits    = CNN_MODEL(tensor)
            cnn_probs     = torch.sigmoid(cnn_logits)

        nsynth_np = nsynth_probs.cpu().numpy()[0]
        cnn_np    = cnn_probs.cpu().numpy()[0]

        # Temporal smoothing on CNN probabilities — mode-isolated buffer
        _prob_history["intermediate"].append(cnn_np)
        if len(_prob_history["intermediate"]) > SMOOTH_WINDOW:
            _prob_history["intermediate"].pop(0)
        smoothed = np.mean(_prob_history["intermediate"], axis=0)

        # Primary note: NSynth winner, with YIN fallback when NSynth is uncertain.
        # Mirrors beginner-mode fusion so a low-confidence NSynth prediction does not
        # silently override a perfectly good YIN pitch estimate.
        nsynth_idx  = int(np.argmax(nsynth_np))
        nsynth_midi = MIDI_MIN + nsynth_idx
        nsynth_conf = float(nsynth_np[nsynth_idx])

        if yin_midi > 0 and abs(yin_midi - nsynth_midi) <= 1:
            fused_midi = yin_midi
            fused_conf = min(0.99, nsynth_conf + 0.05)
        elif nsynth_conf > 0.6:
            fused_midi = nsynth_midi
            fused_conf = nsynth_conf
        elif yin_midi > 0:
            fused_midi = yin_midi
            fused_conf = 0.65
        else:
            fused_midi = nsynth_midi
            fused_conf = nsynth_conf

        return {
            "midi":          fused_midi,
            "note":          NOTE_NAMES[fused_midi % 12],
            "octave":        (fused_midi // 12) - 1,
            "confidence":    round(fused_conf, 4),
            "probabilities": smoothed.tolist(),  # CNN probs for chord detection
            "model_used":    "nsynth+cnn",
        }

    # ── EXPERT / FALLBACK — MAESTRO CNN only ─────────────────────────────────
    # Also used when NSynth model is not loaded (graceful fallback)
    elif CNN_MODEL is not None:
        with torch.no_grad():
            logits = CNN_MODEL(tensor)
            probs  = torch.sigmoid(logits)
        probs_np = probs.cpu().numpy()[0]

        _prob_history["expert"].append(probs_np)
        if len(_prob_history["expert"]) > SMOOTH_WINDOW:
            _prob_history["expert"].pop(0)
        smoothed = np.mean(_prob_history["expert"], axis=0)

        cnn_idx  = int(np.argmax(smoothed))
        cnn_midi = MIDI_MIN + cnn_idx
        cnn_conf = float(smoothed[cnn_idx])

        if yin_midi > 0 and abs(yin_midi - cnn_midi) <= 1:
            fused_midi = yin_midi
            fused_conf = min(0.99, cnn_conf + 0.1)
        elif yin_midi > 0:
            fused_midi = cnn_midi if cnn_conf > 0.6 else yin_midi
            fused_conf = cnn_conf
        else:
            fused_midi = cnn_midi
            fused_conf = cnn_conf

        return {
            "midi":          fused_midi,
            "note":          NOTE_NAMES[fused_midi % 12],
            "octave":        (fused_midi // 12) - 1,
            "confidence":    round(fused_conf, 4),
            "probabilities": smoothed.tolist(),
            "model_used":    "cnn",
        }

    # ── No model available ────────────────────────────────────────────────────
    else:
        raise RuntimeError("No model loaded — start app.py with --model and/or --nsynth_model")


# ─── AUTH ROUTES ──────────────────────────────────────────────────────────────

# Registration route: validates input, checks for duplicates, hashes password, creates user, returns JWT token and user info on success
@app.route("/api/register", methods=["POST"])
def register():
    #GET username, email and password from request body
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    email    = (data.get("email")    or "").strip().lower()
    password =  data.get("password") or ""
    #Validate input
    if not username or not email or not password:
        return jsonify({"error": "username, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 409
    #Hash password and create user
    pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    user    = User(username=username, email=email, password_hash=pw_hash)
    db.session.add(user)
    db.session.commit()
    #Create JWT token and return user info
    token = create_access_token(identity=str(user.id))
    return jsonify({"token": token, "user": user.to_dict()}), 201

#Login route: verifies email and password, returns JWT token and user info on success
@app.route("/api/login", methods=["POST"])
def login():
    #GET email and password from request body
    data     = request.get_json(force=True)
    email    = (data.get("email")    or "").strip().lower()
    password =  data.get("password") or ""
    #Validate input
    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401
    #Create JWT token and return user info
    token = create_access_token(identity=str(user.id))
    return jsonify({"token": token, "user": user.to_dict()}), 200


@app.route("/api/me", methods=["GET"])
@jwt_required()
def me():
    user = User.query.get(int(get_jwt_identity()))
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": user.to_dict()}), 200


@app.route("/api/me/level", methods=["PUT"])
@jwt_required()
def update_level():
    data  = request.get_json(force=True)
    level = (data.get("level") or "").strip().lower()
    if level not in VALID_LEVELS:
        return jsonify({"error": f"Level must be one of: {', '.join(VALID_LEVELS)}"}), 400
    user = User.query.get(int(get_jwt_identity()))
    if not user:
        return jsonify({"error": "User not found"}), 404
    user.level = level
    db.session.commit()
    return jsonify({"user": user.to_dict()}), 200


# ─── MODEL / INFERENCE ROUTES ─────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status":       "ok",
        "model_loaded": CNN_MODEL is not None,
        "model":        MODEL_INFO,
        "nsynth_loaded": NSYNTH_MODEL is not None,
        "nsynth":       NSYNTH_MODEL_INFO,
        "device":       str(DEVICE),
    })


@app.route("/api/predict", methods=["POST"])
@jwt_required()
def predict():
    """
    Accepts:
        { audio: [float...], sr: int, yin_midi: int, level: str }
    level: 'beginner' uses NSynth classifier, 'intermediate'/'expert' uses CNN.
    Returns full probability array so the HTML can do polyphonic thresholding.
    """
    # Require at least one model to be loaded
    if CNN_MODEL is None and NSYNTH_MODEL is None:
        return jsonify({"error": "No model loaded"}), 503
    try:
        data     = request.get_json(force=True)
        audio    = np.array(data["audio"], dtype=np.float32)
        sr       = int(data.get("sr",        44100))
        yin_midi = int(data.get("yin_midi",  0))
        level    = str(data.get("level",     "beginner")).lower()

        # Fallback: if NSynth not loaded, use CNN regardless of level
        if level == "beginner" and NSYNTH_MODEL is None:
            level = "intermediate"

        # Fallback: if CNN not loaded, use NSynth regardless of level
        if level != "beginner" and CNN_MODEL is None:
            level = "beginner"

        result = run_inference(audio, sr, yin_midi, level)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── SESSION + ANALYTICS ROUTES ──────────────────────────────────────────────

@app.route("/api/sessions", methods=["POST"])
@jwt_required()
def save_session():
    """
    Called by the HTML when a song finishes.
    Body: { song_key, song_name, correct, wrong, accuracy,
            timing_avg_ms, duration_secs }
    """
    data = request.get_json(force=True)
    user_id = int(get_jwt_identity())

    session = Session(
        user_id       = user_id,
        song_key      = data.get("song_key",      "unknown"),
        song_name     = data.get("song_name",     "Unknown"),
        correct       = int(data.get("correct",   0)),
        wrong         = int(data.get("wrong",     0)),
        accuracy      = float(data.get("accuracy",0.0)),
        timing_avg_ms = data.get("timing_avg_ms"),   # None is fine
        duration_secs = int(data.get("duration_secs", 0)),
    )
    db.session.add(session)
    db.session.commit()
    return jsonify({"session": session.to_dict()}), 201


@app.route("/api/analytics", methods=["GET"])
@jwt_required()
def analytics():
    """
    Returns everything the analytics page needs in one call:
      - sessions_by_day   : [{date, count, avg_accuracy}] last 30 days
      - streak            : current daily streak (consecutive days with ≥1 session)
      - totals            : total_sessions, total_correct, total_wrong, overall_accuracy
      - song_breakdown    : [{song_name, plays, avg_accuracy}] top songs
      - recent            : last 10 sessions
    """
    from datetime import datetime, timedelta, timezone
    import collections

    user_id  = int(get_jwt_identity())
    all_sess = (Session.query
                .filter_by(user_id=user_id)
                .order_by(Session.played_at.desc())
                .all())

    # ── sessions by day (last 30 days) ────────────────────────────────────────
    today     = datetime.now(timezone.utc).date()
    day_map   = {}   # date_str → {count, acc_sum}
    for s in all_sess:
        if not s.played_at: continue
        d = s.played_at.date()
        if (today - d).days > 29: continue
        key = d.isoformat()
        if key not in day_map: day_map[key] = {"count": 0, "acc_sum": 0.0}
        day_map[key]["count"]   += 1
        day_map[key]["acc_sum"] += s.accuracy

    sessions_by_day = []
    for i in range(29, -1, -1):
        d   = today - timedelta(days=i)
        key = d.isoformat()
        if key in day_map:
            c = day_map[key]["count"]
            sessions_by_day.append({
                "date":         key,
                "count":        c,
                "avg_accuracy": round(day_map[key]["acc_sum"] / c, 1),
            })
        else:
            sessions_by_day.append({"date": key, "count": 0, "avg_accuracy": 0})

    # ── streak ────────────────────────────────────────────────────────────────
    active_dates = sorted({
        s.played_at.date() for s in all_sess if s.played_at
    }, reverse=True)
    streak = 0
    check  = today
    for d in active_dates:
        if d == check or d == check - timedelta(days=1):
            streak += 1
            check   = d
        else:
            break

    # ── totals ────────────────────────────────────────────────────────────────
    total_correct = sum(s.correct  for s in all_sess)
    total_wrong   = sum(s.wrong    for s in all_sess)
    total_notes   = total_correct + total_wrong
    overall_acc   = round(total_correct / total_notes * 100, 1) if total_notes else 0

    # ── song breakdown ────────────────────────────────────────────────────────
    song_map = collections.defaultdict(lambda: {"plays": 0, "acc_sum": 0.0, "song_name": ""})
    for s in all_sess:
        song_map[s.song_key]["plays"]    += 1
        song_map[s.song_key]["acc_sum"]  += s.accuracy
        song_map[s.song_key]["song_name"] = s.song_name
    song_breakdown = sorted([
        {
            "song_key":    k,
            "song_name":   v["song_name"],
            "plays":       v["plays"],
            "avg_accuracy":round(v["acc_sum"] / v["plays"], 1),
        }
        for k, v in song_map.items()
    ], key=lambda x: -x["plays"])[:8]

    return jsonify({
        "sessions_by_day": sessions_by_day,
        "streak":          streak,
        "totals": {
            "sessions":         len(all_sess),
            "correct":          total_correct,
            "wrong":            total_wrong,
            "overall_accuracy": overall_acc,
        },
        "song_breakdown": song_breakdown,
        "recent":         [s.to_dict() for s in all_sess[:10]],
    })


# ─── MIDI REFERENCE ROUTES ────────────────────────────────────────────────────

@app.route("/api/midi-references", methods=["GET"])
@jwt_required()
def get_midi_refs():
    uid  = int(get_jwt_identity())
    refs = MidiReference.query.filter_by(user_id=uid).order_by(MidiReference.id.desc()).all()
    return jsonify([{
        "id":          r.id,
        "title":       r.title,
        "composer":    r.composer,
        "notes_json":  r.notes_json,
        "note_count":  r.note_count,
        "uploaded_at": r.uploaded_at.strftime("%Y-%m-%d") if r.uploaded_at else "",
    } for r in refs])


@app.route("/api/midi-references", methods=["POST"])
@jwt_required()
def save_midi_ref():
    uid  = int(get_jwt_identity())
    data = request.get_json(force=True)
    ref  = MidiReference(
        user_id    = uid,
        title      = (data.get("title") or "Untitled").strip(),
        composer   = (data.get("composer") or "").strip(),
        notes_json = data.get("notes_json", "[]"),
        note_count = int(data.get("note_count", 0)),
    )
    db.session.add(ref)
    db.session.commit()
    return jsonify({"id": ref.id}), 201


@app.route("/api/midi-references/<int:ref_id>", methods=["DELETE"])
@jwt_required()
def delete_midi_ref(ref_id):
    uid = int(get_jwt_identity())
    ref = MidiReference.query.filter_by(id=ref_id, user_id=uid).first_or_404()
    db.session.delete(ref)
    db.session.commit()
    return jsonify({"deleted": True})


# ─── STARTUP ──────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    global CNN_MODEL, DEVICE, MODEL_INFO, BEST_THRESHOLD
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(model_path, map_location=DEVICE, weights_only=False)
    n_mels    = ckpt.get("n_mels",   N_MELS)
    n_frames  = ckpt.get("n_frames", N_FRAMES)
    CNN_MODEL = PianoTranscriptionCNN(n_mels=n_mels, n_frames=n_frames).to(DEVICE)
    CNN_MODEL.load_state_dict(ckpt["model_state"])
    CNN_MODEL.eval()
    MODEL_INFO = {
        "path":     model_path,
        "epoch":    ckpt.get("epoch", "?"),
        "f1":       round(float(ckpt.get("f1", 0)), 4),
        "n_mels":   n_mels,
        "n_frames": n_frames,
        "params":   sum(p.numel() for p in CNN_MODEL.parameters()),
    }
    # Load best threshold saved by phase4_train.py threshold sweep
    thresh_path = Path(model_path).parent / "best_threshold.pt"
    if thresh_path.exists():
        t = torch.load(thresh_path, map_location="cpu", weights_only=False)
        BEST_THRESHOLD = float(t["threshold"])
        print(f"  Best threshold loaded: {BEST_THRESHOLD:.2f}")
    else:
        BEST_THRESHOLD = 0.35
        print(f"  best_threshold.pt not found — using default {BEST_THRESHOLD:.2f}")
    MODEL_INFO["threshold"] = round(BEST_THRESHOLD, 2)
    _prob_history["intermediate"].clear()
    _prob_history["expert"].clear()
    print(f"  Model loaded — epoch {MODEL_INFO['epoch']} · F1 {MODEL_INFO['f1']} · threshold {BEST_THRESHOLD:.2f}")


def load_nsynth_model(model_path: str):
    """Load the NSynth note classifier for beginner mode."""
    global NSYNTH_MODEL, NSYNTH_MODEL_INFO, DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt     = torch.load(model_path, map_location=DEVICE, weights_only=False)
    n_mels   = ckpt.get("n_mels",   N_MELS)
    n_frames = ckpt.get("n_frames", N_FRAMES)
    NSYNTH_MODEL = NSynthPitchCNN(n_mels=n_mels, n_frames=n_frames).to(DEVICE)
    NSYNTH_MODEL.load_state_dict(ckpt["model_state"])
    NSYNTH_MODEL.eval()
    NSYNTH_MODEL_INFO = {
        "path":     model_path,
        "epoch":    ckpt.get("epoch",    "?"),
        "accuracy": round(float(ckpt.get("accuracy", ckpt.get("top1_acc", 0))), 4),
        "loaded":   True,
        "type":     "nsynth_classifier",
    }
    print(f"  NSynth model loaded — epoch {NSYNTH_MODEL_INFO['epoch']} · "
          f"Top-1 acc {NSYNTH_MODEL_INFO['accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Piano Coach server")
    parser.add_argument("--model",        default=None,
        help="Path to best_model.pt (MAESTRO CNN — intermediate/expert)")
    parser.add_argument("--nsynth_model", default=None,
        help="Path to nsynth_best.pt (NSynth classifier — beginner/intermediate)")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()

    print("\n" + "═"*55)
    print("  Piano Coach · Auth + Inference Server")
    print("═"*55)

    with app.app_context():
        db.create_all()
        print("  Database ready (piano_coach.db)")

    if args.model:
        if not Path(args.model).exists():
            print(f"  ⚠  MAESTRO model not found: {args.model}")
        else:
            load_model(args.model)
    else:
        print("  ⚠  No --model given. Intermediate/Expert detection disabled.")

    if args.nsynth_model:
        if not Path(args.nsynth_model).exists():
            print(f"  ⚠  NSynth model not found: {args.nsynth_model}")
        else:
            load_nsynth_model(args.nsynth_model)
    else:
        print("  ⚠  No --nsynth_model given. Beginner mode will use YIN only.")

    print(f"\n  Running on http://localhost:{args.port}")
    print(f"  (also reachable on your local network via the 0.0.0.0 address Flask prints below)")
    print(f"  Open http://localhost:{args.port} in your browser\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
