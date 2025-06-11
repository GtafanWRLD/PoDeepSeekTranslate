
# Deepseek PO Translator

A tool for **batch-translating `.po` localization files** using the Deepseek AI API. Mainly used for Mir Tankov.

---

## ⭐ Features

- **Batch translates .po files** from Russian (or other source languages) to your chosen language using Deepseek AI.
- **Smart detection:** Skips English, empty, numeric, and coordinate entries.
- **Customizable:** Configure batch size, workers, target language, and more.
- **Cache-aware:** Warns and optimizes for Deepseek's discounted hours.
- **API key management:** Will auto-create and open `deepseek_key.txt` if missing.

---

## ⚙️ Configuration

- Settings like `batch_size`, `max_workers`, `api_delay`, and more are stored in `config.json`.  
  First run will prompt for settings, or edit the file to adjust later.

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
