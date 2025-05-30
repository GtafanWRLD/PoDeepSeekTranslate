
# Deepseek PO Translator

A tool for **batch-translating `.po` localization files** using the Deepseek AI API. Mainly used for Mir Tankov.

---

## ⭐ Features

- **Batch translates .po files** from Russian (or other source languages) to your chosen language using Deepseek AI.
- **Multithreaded:** Translates many strings in parallel for speed.
- **Smart detection:** Skips English, empty, numeric, and coordinate entries.
- **Customizable:** Configure batch size, workers, target language, retry count, and more.
- **Cache-aware:** Warns and optimizes for Deepseek's discounted hours.
- **API key management:** Will auto-create and open `deepseek_key.txt` if missing.

---

## 🚀 Usage

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   (Or see `requirements.txt` for needed packages: `deepseek`, `tqdm`, `requests`, etc.)

2. **Get your Deepseek API key** from [deepseek.com](https://deepseek.com/) and paste it into `deepseek_key.txt` (the script will prompt if missing).

3. **Place your `.po` files in the `input/` folder.**

4. **Translated files will appear in the `output/` folder.**

---

## ⚙️ Configuration

- Settings like `batch_size`, `max_workers`, `api_delay`, and more are stored in `config.json`.  
  First run will prompt for settings, or edit the file to adjust later.

---

## 💸 Cost Optimization

- **Runs by default only in Deepseek’s discounted hours (16:00–23:59 UTC).**
- Warns (and allows override) if you try to run outside this window.
- Batches requests for best cache use and minimum cost.

---

## 🧠 How it Works

- Reads all `.po` files from `input/`
- Detects which entries need translation (Russian/Cyrillic or other configurable source)
- Sends strings in batches to Deepseek API
- Writes translated `.po` files to `output/`, keeping original formatting

---

## 🧩 GPT Version

Looking for the **OpenAI GPT-based version** of this project?  
Check out: [GtafanWRLD/PO_AI_Translate](https://github.com/GtafanWRLD/PO_AI_Translate)

---

## ✍️ Author

Made by [GtafanWRLD](https://github.com/GtafanWRLD)  
If you find it useful, feel free to star or contribute!

---

## 📝 License

MIT

---
