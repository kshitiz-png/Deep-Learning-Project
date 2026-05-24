# 🧠 Denoising Autoencoder — Flask Web App
### ULC665 Deep Learning Project

---

## 📁 Project Structure
```
denoising-autoencoder/
├── app.py                  ← Flask server (run this)
├── requirements.txt        ← Python packages
├── weights/
│   └── autoencoder.h5      ← saved model (auto-created on first run)
└── templates/
    └── index.html          ← frontend UI
```

---

## ⚡ Setup & Run (Step by Step)

### Step 1 — Open VS Code terminal
```
Ctrl + ~
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```

### Step 3 — Activate it (Windows)
```bash
venv\Scripts\activate
```

### Step 4 — Install packages
```bash
pip install -r requirements.txt
```

### Step 5 — Run the app
```bash
python app.py
```

### Step 6 — Open browser
```
http://127.0.0.1:5000
```

---

## 🎯 How it Works
1. First run = model trains automatically on MNIST (~2 min)
2. Model weights saved to `weights/autoencoder.h5`
3. Next runs = loads saved weights instantly
4. Upload image OR click Run for random MNIST digit
5. App adds noise → Autoencoder removes noise → Shows before/after

---

## 🛠️ Tech Stack
| Layer     | Technology              |
|-----------|-------------------------|
| Model     | Keras + TensorFlow 2.x  |
| Backend   | Flask                   |
| Frontend  | HTML + CSS + JavaScript |
| Dataset   | MNIST (built into Keras)|

---

## ❗ Common Errors
| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `venv\Scripts\activate` fails | Run VS Code as Administrator |
| Port 5000 busy | Change `app.run(port=5001)` in app.py |
| Slow first run | Normal — model is training, wait ~2 min |
