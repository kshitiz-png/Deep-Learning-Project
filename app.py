from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io, base64, os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.datasets import cifar10

app = Flask(__name__)

IMG_SIZE        = 64
MODEL_PATH      = 'weights/autoencoder_cifar.h5'
ANOMALY_PATH    = 'weights/anomaly_autoencoder.h5'
THRESHOLD_PATH  = 'weights/anomaly_threshold.json'
CIFAR_CLASSES   = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ─── Shared architecture ──────────────────────────────────────────────────────
def build_model():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(32,  (3,3), activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,  (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    out = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    return Model(inp, out)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_gray(imgs):
    out = []
    for img in imgs:
        g = Image.fromarray(img).convert('L').resize((IMG_SIZE, IMG_SIZE))
        out.append(np.array(g))
    return np.array(out, dtype='float32') / 255.0

def prepare_cifar_gray():
    (x_train, _), (x_test, _) = cifar10.load_data()
    print("  Converting CIFAR-10 to grayscale...")
    return (to_gray(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1),
            to_gray(x_test ).reshape(-1, IMG_SIZE, IMG_SIZE, 1))

def to_base64(arr, size=320, colormap=False):
    px = (arr * 255).astype(np.uint8).squeeze()
    if colormap:
        heat = np.zeros((px.shape[0], px.shape[1], 3), dtype=np.uint8)
        heat[:,:,0] = px
        heat[:,:,1] = 255 - px
        img = Image.fromarray(heat, mode='RGB').resize((size, size), Image.LANCZOS)
    else:
        img = Image.fromarray(px, mode='L').resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 1 — DENOISING AUTOENCODER
# ════════════════════════════════════════════════════════════════════════════════
if os.path.exists(MODEL_PATH):
    print("Loading denoising model...")
    denoise_model = build_model()
    denoise_model.load_weights(MODEL_PATH)
else:
    print("Training denoising model (~5 min)...")
    x_train, x_test = prepare_cifar_gray()
    noise  = 0.35
    x_tr_n = np.clip(x_train + noise * np.random.randn(*x_train.shape), 0, 1)
    x_te_n = np.clip(x_test  + noise * np.random.randn(*x_test.shape),  0, 1)
    denoise_model = build_model()
    denoise_model.compile(optimizer='adam', loss='binary_crossentropy')
    denoise_model.fit(x_tr_n, x_train, epochs=15, batch_size=64,
                      validation_data=(x_te_n, x_test), verbose=1)
    os.makedirs('weights', exist_ok=True)
    denoise_model.save_weights(MODEL_PATH)
    print("Denoising model saved!")

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 2 — ANOMALY DETECTION AUTOENCODER
# Trained ONLY on "automobile" images (class 1)
# High reconstruction error = anomaly | Low error = normal (car-like)
# ════════════════════════════════════════════════════════════════════════════════
if os.path.exists(ANOMALY_PATH) and os.path.exists(THRESHOLD_PATH):
    print("Loading anomaly detection model...")
    anomaly_model = build_model()
    anomaly_model.load_weights(ANOMALY_PATH)
    with open(THRESHOLD_PATH) as f:
        d = json.load(f)
        ANOMALY_THRESHOLD = d['threshold']
        NORMAL_LABEL      = d['normal_class']
    print(f"  Threshold={ANOMALY_THRESHOLD:.4f}  Normal class={NORMAL_LABEL}")
else:
    print("Training anomaly model on 'automobile' class only (~3 min)...")
    (x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw) = cifar10.load_data()
    tr_mask = (y_tr_raw.flatten() == 1)
    te_mask = (y_te_raw.flatten() == 1)
    x_anom_tr = to_gray(x_tr_raw[tr_mask]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_anom_te = to_gray(x_te_raw[te_mask]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print(f"  {len(x_anom_tr)} normal (automobile) training images")
    anomaly_model = build_model()
    anomaly_model.compile(optimizer='adam', loss='mse')
    anomaly_model.fit(x_anom_tr, x_anom_tr, epochs=20, batch_size=32,
                      validation_data=(x_anom_te, x_anom_te), verbose=1)
    recon  = anomaly_model.predict(x_anom_tr, verbose=0)
    errors = np.mean(np.square(x_anom_tr - recon), axis=(1,2,3))
    ANOMALY_THRESHOLD = float(np.mean(errors) + 2 * np.std(errors))
    NORMAL_LABEL      = 'automobile'
    os.makedirs('weights', exist_ok=True)
    anomaly_model.save_weights(ANOMALY_PATH)
    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold': ANOMALY_THRESHOLD, 'normal_class': NORMAL_LABEL}, f)
    print(f"Anomaly model saved!  Threshold={ANOMALY_THRESHOLD:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/denoise', methods=['POST'])
def denoise():
    try:
        file        = request.files.get('image')
        noise_level = float(request.form.get('noise_level', 0.35))
        if file:
            pil = Image.open(file).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            clean = np.array(pil, dtype='float32') / 255.0
            clean = clean.reshape(IMG_SIZE, IMG_SIZE, 1)
        else:
            (_, _), (x_test, _) = cifar10.load_data()
            idx = np.random.randint(0, len(x_test))
            pil = Image.fromarray(x_test[idx]).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            clean = np.array(pil, dtype='float32') / 255.0
            clean = clean.reshape(IMG_SIZE, IMG_SIZE, 1)
        noisy    = np.clip(clean + noise_level * np.random.randn(*clean.shape), 0, 1)
        denoised = denoise_model.predict(noisy.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
        return jsonify({'original': to_base64(clean), 'noisy': to_base64(noisy),
                        'denoised': to_base64(denoised), 'status': 'success'})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/anomaly', methods=['POST'])
def anomaly():
    try:
        file       = request.files.get('image')
        true_class = 'uploaded image'
        if file:
            pil     = Image.open(file).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img_arr = np.array(pil, dtype='float32') / 255.0
            img_arr = img_arr.reshape(IMG_SIZE, IMG_SIZE, 1)
        else:
            (_, _), (x_te_raw, y_te_raw) = cifar10.load_data()
            idx        = np.random.randint(0, len(x_te_raw))
            pil        = Image.fromarray(x_te_raw[idx]).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img_arr    = np.array(pil, dtype='float32') / 255.0
            img_arr    = img_arr.reshape(IMG_SIZE, IMG_SIZE, 1)
            true_class = CIFAR_CLASSES[int(y_te_raw[idx])]

        reconstructed = anomaly_model.predict(img_arr.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
        error_map     = np.square(img_arr - reconstructed)
        recon_error   = float(np.mean(error_map))
        error_visual  = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)

        is_anomaly  = recon_error > ANOMALY_THRESHOLD
        verdict     = 'ANOMALY' if is_anomaly else 'NORMAL'
        ratio       = recon_error / ANOMALY_THRESHOLD
        confidence  = min(99, int(abs(ratio - 1) * 100 + 50)) if is_anomaly else min(99, int((1 - ratio) * 100 + 55))

        return jsonify({
            'original':      to_base64(img_arr),
            'reconstructed': to_base64(reconstructed),
            'error_map':     to_base64(error_visual, colormap=True),
            'recon_error':   round(recon_error * 1000, 3),
            'threshold':     round(ANOMALY_THRESHOLD * 1000, 3),
            'verdict':       verdict,
            'confidence':    confidence,
            'normal_class':  NORMAL_LABEL,
            'true_class':    true_class,
            'status':        'success'
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
