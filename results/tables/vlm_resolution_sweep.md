| render size | processor setting | image tokens / image | accuracy | ms / question | peak alloc GB |
|---|---|---|---|---|---|
| 128 | model-card defaults | 64.0 | 0.833 | 17.5 | 1.087 |
| 256 | model-card defaults | 64.0 | 0.844 | 17.4 | 1.087 |
| 512 | model-card defaults | 256.0 | 0.889 | 23.5 | 1.079 |
| 768 | model-card defaults | 1280.0 | 0.811 | 117.8 | 1.734 |
| 1024 | model-card defaults | 1280.0 | 0.811 | 118.2 | 1.734 |
| 512 | {'max_image_tokens': 64} | 1088.0 | 0.878 | 115.9 | 1.776 |
| 512 | {'max_image_tokens': 128} | 121.0 | 0.878 | 20.0 | 1.087 |
| 512 | {'max_image_tokens': 64, 'do_image_splitting': False} | 64.0 | 0.867 | 8.5 | 0.963 |
| 1024 | {'do_image_splitting': False} | 256.0 | 0.811 | 24.9 | 1.079 |
| 1024 | {'max_image_tokens': 64, 'do_image_splitting': False} | 64.0 | 0.867 | 9.5 | 0.963 |

Base LFM2.5-VL-450M, greedy, batch 8, 90 test questions, single run. Latency was measured while other GPU/CPU jobs shared the laptop, so treat it as indicative.
