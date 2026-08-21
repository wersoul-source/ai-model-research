# รายงาน tokenizer_config.json — Qwen3-4B

> ไฟล์: `/root/models/Qwen3-4B/tokenizer_config.json` (9.7 K) | ตรวจ: 63/63 PASS

## เนื้อหาสำคัญ

- **tokenizer_class:** Qwen2Tokenizer
- **model_max_length:** 131072 (YaRN max)
- **bos_token:** null, eos `<|im_end|>` (151645), pad `<|endoftext|>` (151643)
- **added_tokens_decoder:** 26 entries (151643-151668)
- **chat_template:** Jinja2 ยาว ~4K chars — รองรับ enable_thinking, tool_call, multi-step tool

## Chat Template (สรุป)

```jinja
{% if tools %} <|im_start|>system ... # Tools ... <tools> ... {% endif %}
{% for message in messages %}
  {% if user %} <|im_start|>user\n{content}<|im_end|>
  {% elif assistant %}
    {% set reasoning = message.reasoning_content or split(content,'</think>') %}
    {% if enable_thinking %} <think>{reasoning}</think> {% endif %}
    {% if tool_calls %} <tool_call>{"name":..., "arguments":...}</tool_call> {% endif %}
  {% elif tool %} <|im_start|>user\n<tool_response>{content}</tool_response><|im_end|>
{% endfor %}
{% if enable_thinking==false %} <think>\n\n</think>\n\n {% endif %}
```

## Thinking Mode Switching

- **enable_thinking=True (default):** output `<think>reasoning</think>\n\nanswer`
- **enable_thinking=False:** output `answer` (template ใส่ empty think block ให้)
- **Soft switch:** `/think` / `/no_think` ใน user input (last flag wins)
- **Thinking budget:** หยุดที่ threshold → insert `Considering limited time...`

## เทียบ Gemma 4 E4B-it

| Qwen3-4B | Gemma 4 E4B-it |
|----------|----------------|
| `<think>` / `</think>` pair (151667/151668) | `<|think|>` + `<|channel>thought` |
| enable_thinking flag | enable_thinking flag |
| /think /no_think in input | ไม่มี soft switch |
| tool_call JSON in template | tool_call JSON in jinja (คล้ายกัน) |

*ตรวจด้วย diagnose-qwen3.py: chat_template checks 2/2 PASS (enable_thinking + tool_call)*
