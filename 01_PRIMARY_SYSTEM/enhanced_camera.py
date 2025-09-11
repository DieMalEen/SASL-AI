# ==== SASL-AI Enhanced Camera + Grammar & JSON Save (drop-in) ====
# Suppress noisy logs (MediaPipe / TF)
import os as _os, sys as _sys, warnings as _warnings
_os.environ['GLOG_minloglevel'] = '2'
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
_warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

# ---- Standard imports
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import deque

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ---- Optional hand detector
try:
    from hand_detection import HandDetector
    HAND_DETECTION_AVAILABLE = True
except Exception:
    HAND_DETECTION_AVAILABLE = False

# =========================
#   PATHS & CLASS NAMES
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
data_cfg_dir = os.path.join(project_root, "03_DATA_CONFIG")

def load_class_names() -> List[str]:
    path = os.path.join(data_cfg_dir, "class_names.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not classes:
            raise ValueError("class_names.json must be a non-empty JSON array.")
        print(f"Loaded {len(classes)} classes.")
        return classes
    except Exception as e:
        print(f"ERROR loading class_names.json at {path}: {e}")
        sys.exit(1)

class_names = load_class_names()

# =========================
#   CNN BASE (from repo)
# =========================
# Try to import cnn_base from your hand_focused_CNN_LSTM, else fallback model.py
import importlib.util as _ilu
cnn_base = None
try:
    _p = os.path.join(current_dir, "hand_focused_CNN_LSTM.py")
    _spec = _ilu.spec_from_file_location("hand_focused_CNN_LSTM", _p)
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)  # type: ignore
    cnn_base = _mod.cnn_base
    print("cnn_base: hand_focused_CNN_LSTM")
except Exception:
    try:
        _p = os.path.join(project_root, "02_FALLBACK_COMPATIBILITY", "model.py")
        _spec = _ilu.spec_from_file_location("model", _p)
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)  # type: ignore
        cnn_base = _mod.cnn_base
        print("cnn_base: fallback model")
    except Exception as e:
        print("Could not load cnn_base:", e)
        sys.exit(1)

# =========================
#   MODEL AUTO-LOADER
# =========================
MODEL_TYPE = None
model = None

def _validate_model(_model) -> bool:
    try:
        x = torch.randn(1, 16, 3, 224, 224).to(device)
        with torch.no_grad():
            y = _model(x)
            if torch.isnan(y).any() or torch.isinf(y).any():
                return False
            p = torch.softmax(y, dim=1)
            if torch.isnan(p).any() or torch.isinf(p).any():
                return False
        return True
    except Exception:
        return False

def _load_by_arch(path: str, state):
    global model, MODEL_TYPE
    # Identify architecture by keys
    keys = list(state.keys())
    def has(k): return any(k in kk for kk in keys)
    try:
        if not has("lstm") and not has("attention"):
            # FastCNNLSTM from gpu_optimized_training
            p = os.path.join(current_dir, "gpu_optimized_training.py")
            spec = _ilu.spec_from_file_location("gpu_optimized_training", p)
            mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
            model = mod.FastCNNLSTM(num_classes=len(class_names)).to(device)
            model.load_state_dict(state)
            MODEL_TYPE = "gpu_optimized"
        else:
            # Hand-focused or Hybrid depending on filename
            if "hybrid" in os.path.basename(path).lower():
                p = os.path.join(current_dir, "hybrid_cpu_gpu_training.py")
                spec = _ilu.spec_from_file_location("hybrid_cpu_gpu_training", p)
                mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
                model = mod.HybridOptimizedCNN_LSTM(cnn=cnn_base, num_classes=len(class_names),
                                                    hidden_size=256, num_layers=2, dropout=0.3).to(device)
                model.load_state_dict(state)
                MODEL_TYPE = "hybrid_cpu_gpu"
            else:
                p = os.path.join(current_dir, "hand_focused_CNN_LSTM.py")
                spec = _ilu.spec_from_file_location("hand_focused_CNN_LSTM", p)
                mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
                model = mod.HandFocusedCNN_LSTM(cnn=cnn_base, num_classes=len(class_names),
                                                hidden_size=256, num_layers=2, dropout=0.3).to(device)
                model.load_state_dict(state)
                MODEL_TYPE = "hand_focused"
        if _validate_model(model):
            return True
    except Exception as e:
        print("Load-by-arch failed:", e)
    model = None; MODEL_TYPE = None
    return False

# Forced path (optional)
forced_model_path = os.environ.get("SASL_FORCE_MODEL_PATH")
cand_paths = [
    os.path.join(output_dir, "gpu_sasl_model.pth"),
    os.path.join(output_dir, "gpu_optimized_sasl_model.pth"),
    os.path.join(output_dir, "hybrid_sasl_model.pth"),
    os.path.join(output_dir, "hybrid_cpu_gpu_sasl_model.pth"),
    os.path.join(output_dir, "hand_focused_sasl_model.pth"),
    "gpu_sasl_model.pth","gpu_optimized_sasl_model.pth",
    "hybrid_sasl_model.pth","hybrid_cpu_gpu_sasl_model.pth","hand_focused_sasl_model.pth",
]

if forced_model_path and os.path.exists(forced_model_path):
    try:
        st = torch.load(forced_model_path, map_location=device, weights_only=False)
        if _load_by_arch(forced_model_path, st):
            print("Loaded forced model:", forced_model_path)
    except Exception as e:
        print("Forced model error:", e)

if model is None:
    for p in cand_paths:
        if os.path.exists(p):
            try:
                st = torch.load(p, map_location=device, weights_only=False)
                if _load_by_arch(p, st):
                    print("Loaded model:", p)
                    break
            except Exception as e:
                print("Skip", p, "→", e)

if model is None:
    # last resort: untrained standard CNN_LSTM
    try:
        p = os.path.join(project_root, "02_FALLBACK_COMPATIBILITY", "model.py")
        spec = _ilu.spec_from_file_location("model", p)
        mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
        model = mod.CNN_LSTM(cnn=cnn_base, num_classes=len(class_names)).to(device)
        MODEL_TYPE = "standard"
        print("Using standard CNN-LSTM (no weights)")
    except Exception as e:
        print("ERROR: cannot construct any model:", e)
        sys.exit(1)

model.eval()
if MODEL_TYPE is None: MODEL_TYPE = "unknown"
print("Model ready:", MODEL_TYPE)

# =========================
#   TRANSFORMS
# =========================
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def _pre(frames: List[np.ndarray]) -> torch.Tensor:
    return torch.stack([_transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames])

# =========================
#   HAND ROI (optional)
# =========================
def extract_hand_region_from_frame(frame: np.ndarray, hand_detector) -> np.ndarray:
    if not HAND_DETECTION_AVAILABLE or hand_detector is None: return frame
    try:
        hands = hand_detector.detect_hands(frame)
        if hands:
            best = max(hands, key=lambda x: x['confidence'])
            x0,y0,x1,y1 = best['bbox']
            x0=max(0,x0); y0=max(0,y0); x1=min(frame.shape[1],x1); y1=min(frame.shape[0],y1)
            crop = frame[y0:y1, x0:x1]
            if crop.size>0:
                return cv2.resize(crop, (frame.shape[1], frame.shape[0]))
    except Exception as e:
        pass
    return frame

# =========================
#   DETECTOR WRAPPER
# =========================
class EnhancedGestureDetector:
    def __init__(self, buffer_size=16, stability_threshold=2, use_hand_detection=True):
        self.frame_buffer = deque(maxlen=buffer_size)
        self.prediction_history = deque(maxlen=stability_threshold)
        self.use_hand_detection = bool(use_hand_detection and HAND_DETECTION_AVAILABLE)
        self.hand_detector = None
        if self.use_hand_detection:
            try:
                self.hand_detector = HandDetector(static_image_mode=False, max_num_hands=2,
                                                  min_detection_confidence=0.5, min_tracking_confidence=0.3)
                print("Hand detection: ON")
            except Exception as e:
                print("Hand detector init failed:", e)
                self.use_hand_detection = False

    def add_frame(self, frame: np.ndarray):
        if self.use_hand_detection and self.hand_detector:
            frame = extract_hand_region_from_frame(frame, self.hand_detector)
        self.frame_buffer.append(frame)

    def is_ready(self) -> bool:
        return len(self.frame_buffer) == self.frame_buffer.maxlen

    def predict(self) -> Tuple[Optional[str], float]:
        if not self.is_ready(): return None, 0.0
        try:
            x = _pre(list(self.frame_buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                y = model(x)
                if torch.isnan(y).any() or torch.isinf(y).any(): return None, 0.0
                p = torch.softmax(y, dim=1)
                conf, idx = torch.max(p, 1)
                idx_i = int(idx.item())
                if idx_i < 0 or idx_i >= len(class_names): return None, 0.0
                cls = class_names[idx_i]
                score = float(conf.item())
                if score < 0.01: return None, 0.0
                return cls, score
        except Exception as e:
            return None, 0.0

    def overlays(self, frame):
        if not (self.use_hand_detection and self.hand_detector): return frame
        try:
            hands = self.hand_detector.detect_hands(frame)
            return self.hand_detector.draw_hands(frame, hands, True, True)
        except Exception:
            return frame

    def cleanup(self):
        pass

# =========================
#   COMPREHENSIVE GRAMMAR
# =========================
V_TIME_FUTURE = {"tomorrow","next","later","soon","tonight","this_evening","this_afternoon"}
V_TIME_PAST   = {"yesterday","ago","earlier"}
TIME_WORDS    = {"today","now","morning","afternoon","evening","tonight","week","month","year",
                 "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
                 "tomorrow","yesterday","later","soon","next","last","ago","earlier","today"}
NEG_WORDS     = {"not","no","dont","don't","cant","can't","wont","won't","never","nothing","none"}
Q_WORDS       = {"who","what","when","where","why","how","whom","which"}
DEST_NOUNS_TO = {"school","work","home","office","shop","store","church","bank","clinic","hospital",
                 "class","lecture","campus","gym","library"}
PLURALS       = {"books":"book","cars":"car","hands":"hand","days":"day","weeks":"week","months":"month","years":"year"}

PRONOUNS = {"i":"I","you":"you","he":"he","she":"she","they":"they","we":"we","me":"I","him":"he","her":"she","them":"they","us":"we"}

A_AN_SET = {"coffee","apple","idea","appointment","car","meeting","book","message","question"}

def _is_vowel(x:str)->bool: return x[:1].lower() in "aeiou"

def a_an(word:str)->str:
    if not word: return word
    return ("an " if _is_vowel(word) else "a ") + word

def progressive(verb:str)->str:
    v = verb.lower()
    if v.endswith("ie"): return v[:-2]+"ying"
    if v.endswith("e") and v not in {"be","see","flee","knee","tie"}: return v[:-1]+"ing"
    if len(v)>=3 and v[-1] not in "aeiou" and v[-2] in "aeiou" and v[-3] not in "aeiou":
        return v + v[-1] + "ing"
    return v + "ing"

def past_simple(verb:str)->str:
    v = verb.lower()
    irr = {"go":"went","see":"saw","be":"was","eat":"ate","meet":"met","have":"had","do":"did","make":"made","come":"came","say":"said","get":"got","buy":"bought","teach":"taught","leave":"left","bring":"brought","run":"ran","write":"wrote","read":"read"}
    if v in irr: return irr[v]
    if v.endswith("e"): return v + "d"
    if len(v)>=3 and v[-1] not in "aeiou" and v[-2] in "aeiou" and v[-3] not in "aeiou":
        return v + v[-1] + "ed"
    return v + "ed"

def choose_subject(tokens: List[str])->str:
    # First explicit pronoun wins
    for t in tokens:
        lt = t.lower()
        if lt in PRONOUNS: return PRONOUNS[lt]
    return "I"  # default speaker assumption

def detect_question(tokens: List[str])->bool:
    return any(t.lower() in Q_WORDS for t in tokens)

def detect_negation(tokens: List[str])->bool:
    return any(t.lower() in NEG_WORDS for t in tokens)

def split_time_tokens(tokens: List[str])->Tuple[List[str], List[str]]:
    times = []
    rest = []
    for t in tokens:
        if t.lower() in TIME_WORDS: times.append(t.lower())
        else: rest.append(t)
    return times, rest

def map_destinations(words: List[str])->List[str]:
    out=[]
    for w in words:
        lw = w.lower()
        if lw in DEST_NOUNS_TO: out.extend(["to", lw] if lw not in {"home","work"} else [lw])  # "go home" (no 'to')
        else: out.append(lw)
    return out

def insert_articles(words: List[str])->List[str]:
    out=[]
    for i,w in enumerate(words):
        lw = w.lower()
        if lw in A_AN_SET:
            # if already has determiner before, skip
            if i>0 and words[i-1].lower() in {"a","an","the","my","your","his","her","their","our"}:
                out.append(lw)
            else:
                out.append("an" if _is_vowel(lw) else "a"); out.append(lw)
        else:
            out.append(lw)
    return out

def plural_to_singular(w:str)->str:
    return PLURALS.get(w.lower(), w.lower())

def tidy(s:str)->str:
    s = s.strip()
    s = s.replace("  "," ")
    if not s: return s
    s = s[0].upper()+s[1:]
    if s[-1] not in ".?!": s += "."
    return s

def grammar_fix(words_in: List[str]) -> str:
    """
    Rule-based SASL→English:
    - pronoun inference
    - tense from time words (past/yesterday, future/tomorrow)
    - negation (don't/can't)
    - progressive 'am V-ing' for present/near future
    - destination preps 'to/at/in/on'
    - articles a/an
    - simple question shaping with wh-words
    """
    if not words_in: return ""

    # Normalize and light clean
    toks = [t for t in (w.strip().lower().replace("_"," ") for w in words_in) if t]
    # merge multi-word time hints
    toks = [("this_evening" if t=="this evening" else "this_afternoon" if t=="this afternoon" else t) for t in toks]

    # base subject
    subj = choose_subject(toks)

    # detect flags
    is_question = detect_question(toks)
    is_neg = detect_negation(toks)
    time_tokens, core = split_time_tokens(toks)

    # light destination & articles
    core = map_destinations(core)
    core = insert_articles(core)

    # pull simple verb if present
    verb = None
    rest = []
    if core:
        verb = core[0]
        rest = core[1:]

    # tiny phrase library (high confidence shortcuts)
    hard_rules = [
        ({"work","tomorrow"}, f"{subj} will work tomorrow"),
        ({"work","today"},    f"{subj} am {progressive('work')} today"),
        ({"go","home"},       f"{subj} am {progressive('go')} home"),
        ({"meet","you","tomorrow"}, f"{subj} will meet you tomorrow"),
        ({"want","coffee"},   f"{subj} want {('an' if _is_vowel('coffee') else 'a')} coffee"),
        ({"call","you","tomorrow"}, f"{subj} will call you tomorrow"),
        ({"eat","lunch","today"}, f"{subj} am {progressive('eat')} lunch today"),
        ({"study","tonight"}, f"{subj} will study tonight"),
    ]
    set_toks = set(toks)
    for need, out in hard_rules:
        if need.issubset(set_toks):
            return tidy(out)

    # fallback synthesis
    # tense from time tokens
    tense = "present"
    if any(t in V_TIME_PAST for t in time_tokens):
        tense = "past"
    elif any(t in V_TIME_FUTURE for t in time_tokens) or "tomorrow" in time_tokens:
        tense = "future"

    # choose verb if missing
    if verb is None:
        # If only noun given: "coffee" -> "I want a coffee"
        if rest:
            first_n = rest[0]
            phrase = f"{subj} want {first_n}"
            if len(rest) > 1:
                phrase += " " + " ".join(rest[1:])
            # add time tokens later
        else:
            phrase = subj
    else:
        # conj by tense and neg
        if is_question:
            # wh-question or yes/no
            wh = next((t for t in toks if t in Q_WORDS), None)
            if wh:
                # where go school -> "Where am I going to school?"
                v = progressive(verb) if tense!="past" else verb
                aux = "am" if subj=="I" else "are"
                phrase = f"{wh.capitalize()} {aux} {subj} {v}"
            else:
                # yes/no → "Do I VERB ... ?"
                base = verb
                aux = "Did" if tense=="past" else ("Will" if tense=="future" else "Do")
                phrase = f"{aux} {subj} {base}"
        else:
            if tense == "past":
                v = past_simple(verb)
                if is_neg:
                    # didn't + base
                    v = f"did not {verb}"
                phrase = f"{subj} {v}"
            elif tense == "future":
                v = verb
                if is_neg:
                    phrase = f"{subj} will not {v}"
                else:
                    phrase = f"{subj} will {v}"
            else:
                # present/progressive default
                if verb in {"be"}:
                    phrase = f"{subj} am" if subj=="I" else f"{subj} is"
                else:
                    v = progressive(verb)
                    aux = "am" if subj=="I" else "are"
                    if is_neg:
                        phrase = f"{subj} {aux} not {v}"
                    else:
                        phrase = f"{subj} {aux} {v}"

    # attach rest tokens neatly
    if rest:
        # remove duplicate determiners like "... a a ..." just in case
        cleaned=[]
        for tok in rest:
            if cleaned and tok in {"a","an","the"} and cleaned[-1] in {"a","an","the"}:
                continue
            cleaned.append(tok)
        rest = cleaned

        # reposition "to" if missing for destinations: already handled in map_destinations
        phrase += " " + " ".join(rest)

    # put time at end (natural English order)
    if time_tokens:
        # normalize some
        tt = []
        for t in time_tokens:
            if t == "this_evening": tt.append("this evening")
            elif t == "this_afternoon": tt.append("this afternoon")
            else: tt.append(t)
        phrase += " " + " ".join(tt)

    return tidy(phrase)

# =========================
#   JSON SAVE
# =========================
def save_session_json(out_dir: str, words: List[Dict[str, float]], sentence: str, meta: Dict[str, str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "session_transcripts.json")
    session_obj = {
        "session_id": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "model_type": meta.get("model_type", ""),
        "words": words,           # [{"text": "...", "confidence": 0.93, "t_utc": "..."}]
        "final_sentence": sentence
    }
    try:
        data = json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []
        if not isinstance(data, list): data = []
    except Exception:
        data = []
    data.append(session_obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

# =========================
#   CAMERA LOOP
# =========================
def run_enhanced_camera():
    print("Starting Enhanced SASL Camera")
    print(f"Model: {MODEL_TYPE}  |  Hand: {'ON' if HAND_DETECTION_AVAILABLE else 'OFF'}")
    print("Controls: q=quit, h=toggle hands, r=reset, u=undo, s=save")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not available")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = EnhancedGestureDetector(buffer_size=8, stability_threshold=2, use_hand_detection=HAND_DETECTION_AVAILABLE)
    show_hand_overlay = True

    # Session state
    recognized_words: List[Dict[str, float]] = []
    last_display_word = None
    last_display_start = None
    last_committed_word = None
    COMMIT_SECONDS = 0.5

    while True:
        ok, frame = cap.read()
        if not ok: break

        detector.add_frame(frame.copy())
        display = frame.copy()

        # Overlay (skeleton/boxes)
        if show_hand_overlay:
            display = detector.overlays(display)

        if detector.is_ready():
            pred, conf = detector.predict()

            if pred:
                # You already draw the main word; we still show a helpful timer + committed list
                cv2.putText(display, f"Gesture: {pred} ({conf:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                # Hold-to-commit
                now = time.monotonic()
                if pred != last_display_word:
                    last_display_word = pred
                    last_display_start = now
                shown_for = 0.0 if last_display_start is None else (now - last_display_start)

                cv2.putText(display, f"Hold-to-commit: {shown_for:.1f}/{COMMIT_SECONDS:.1f}s",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

                if shown_for >= COMMIT_SECONDS and pred != last_committed_word:
                    recognized_words.append({
                        "text": pred,
                        "confidence": round(float(conf), 4),
                        "t_utc": datetime.utcnow().isoformat() + "Z",
                    })
                    last_committed_word = pred
                    last_display_start = now  # allow repeat if kept longer

            # Show committed stream + live sentence preview
            committed = [w["text"] for w in recognized_words]
            cv2.putText(display, "Committed: " + (" ".join(committed) if committed else "—"),
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            preview_sentence = grammar_fix(committed)
            if preview_sentence:
                cv2.putText(display, "Sentence: " + preview_sentence,
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2)
        else:
            cv2.putText(display, f"Collecting frames: {len(detector.frame_buffer)}/{detector.frame_buffer.maxlen}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # Footer controls
        cv2.putText(display, "q:quit  h:hands  r:reset  u:undo  s:save",
                    (10, display.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Enhanced SASL Recognition", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('h'):
            show_hand_overlay = not show_hand_overlay
            print("Hand overlay:", "ON" if show_hand_overlay else "OFF")
        elif key == ord('r'):
            detector.frame_buffer.clear()
            detector.prediction_history.clear()
            recognized_words.clear()
            last_committed_word = last_display_word = None
            last_display_start = None
            print("Reset")
        elif key == ord('u'):
            if recognized_words:
                removed = recognized_words.pop()
                print("Undo:", removed["text"])
        elif key == ord('s'):
            sentence = grammar_fix([w["text"] for w in recognized_words])
            path = save_session_json(output_dir, recognized_words, sentence, {"model_type": MODEL_TYPE})
            print(f"Saved: {path}\n→ {sentence}")

    # Auto-save on exit
    if recognized_words:
        sentence = grammar_fix([w["text"] for w in recognized_words])
        saved_path = save_session_json(output_dir, recognized_words, sentence, {"model_type": MODEL_TYPE})
        print(f"[AUTO-SAVE] {saved_path}")
        print(f"[SENTENCE]  {sentence}")

    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
    print("Camera session ended")

if __name__ == "__main__":
    run_enhanced_camera()
