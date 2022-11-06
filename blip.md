# BLIP

## Loading

Models available: `BLIP` and `BLIP_Large`

```
from nataili.model_manager import ModelManager
mm = ModelManager()
mm.init()

success = mm.load_model("BLIP")

success = mm.load_model("BLIP_Large")
```

## Usage

```
from nataili.blip.caption import Caption

image = PIL.Image.open('image.png').convert('RGB')

blip = Caption(mm.loaded_models["BLIP"]["model"], mm.loaded_models["BLIP"]["device"])

blip_large = Caption(mm.loaded_models["BLIP_Large"]["model"], mm.loaded_models["BLIP_Large"]["device"])

caption = blip(image)
caption_large = blip_large(image)
```

## Options

```
caption = blip(image, sample=True, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0)
```

## test_caption.py output

```
INFO       | 2022-11-06 17:05:00 | __main__:test_caption:20 - Loading BLIP
reshape position embedding from 576 to 1024
load checkpoint from models/blip/model__base_caption.pth
INIT       | 1           | Loading BLIP
INIT       | 1           | Loading BLIP: Took 22.85970115661621 seconds
INFO       | 2022-11-06 17:05:23 | __main__:test_caption:31 - Fast test for BLIP
INFO       | 2022-11-06 17:05:36 | __main__:test_caption:32 - caption: a digital painting of a woman with headphones sample: False
INFO       | 2022-11-06 17:05:37 | __main__:test_caption:33 - caption: there is a digital art image of a person with headphones on sample: True
INFO       | 2022-11-06 17:05:37 | __main__:test_caption:56 - Total time: 14.059289455413818
INFO       | 2022-11-06 17:05:37 | __main__:test_caption:20 - Loading BLIP_Large
reshape position embedding from 576 to 1024
load checkpoint from models/blip/model_large_caption.pth
INIT       | 1           | Loading BLIP_Large
INIT       | 1           | Loading BLIP_Large: Took 34.84111189842224 seconds
INFO       | 2022-11-06 17:06:12 | __main__:test_caption:31 - Fast test for BLIP_Large
INFO       | 2022-11-06 17:06:13 | __main__:test_caption:32 - caption: a drawing of a woman with headphones on sample: False
INFO       | 2022-11-06 17:06:13 | __main__:test_caption:33 - caption: a picture of an illustration of a woman sample: True
INFO       | 2022-11-06 17:06:13 | __main__:test_caption:56 - Total time: 1.6920056343078613
INFO       | 2022-11-06 17:06:13 | __main__:test_caption:35 - Slow test for BLIP
INFO       | 2022-11-06 17:06:14 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:14 | __main__:test_caption:46 - an illustrated image of a young woman with headphones on
INFO       | 2022-11-06 17:06:15 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:15 | __main__:test_caption:46 - the artwork shows a woman with earbuds in her ears
INFO       | 2022-11-06 17:06:16 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:16 | __main__:test_caption:46 - a woman in yellow is standing in the middle of a portrait
INFO       | 2022-11-06 17:06:17 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:17 | __main__:test_caption:46 - a portrait that was painted from an image of a young person wearing a headset
INFO       | 2022-11-06 17:06:18 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:18 | __main__:test_caption:46 - a young girl with headphones and a collar
INFO       | 2022-11-06 17:06:19 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:19 | __main__:test_caption:46 - the woman with a black bow tie is dressed in headphones
INFO       | 2022-11-06 17:06:20 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:20 | __main__:test_caption:46 - this portrait of a girl is being made into a digital artwork
INFO       | 2022-11-06 17:06:21 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:21 | __main__:test_caption:46 - this is an artistic image of a woman with headphones
INFO       | 2022-11-06 17:06:21 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:21 | __main__:test_caption:46 - a drawing shows a woman in headphones
INFO       | 2022-11-06 17:06:22 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:22 | __main__:test_caption:46 - an artistic piece with brown and yellow colors
INFO       | 2022-11-06 17:06:23 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:23 | __main__:test_caption:46 - there is a woman dressed in various colors and hair
INFO       | 2022-11-06 17:06:23 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:23 | __main__:test_caption:46 - the illustration shows a girl in headphones
INFO       | 2022-11-06 17:06:24 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:24 | __main__:test_caption:46 - a digital painting of a young woman in headphones
INFO       | 2022-11-06 17:06:25 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:25 | __main__:test_caption:46 - a portrait that is made with different objects
INFO       | 2022-11-06 17:06:25 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:25 | __main__:test_caption:46 - some sort of artistic portrait painted using head phones
INFO       | 2022-11-06 17:06:26 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:26 | __main__:test_caption:46 - a watercolouric painting of a woman in a yellow shirt
INFO       | 2022-11-06 17:06:27 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:27 | __main__:test_caption:46 - an image of an artistic woman wearing headphones
INFO       | 2022-11-06 17:06:27 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:27 | __main__:test_caption:46 - a girl in headphones has a paint splattered effect
INFO       | 2022-11-06 17:06:28 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:28 | __main__:test_caption:46 - a digital drawing of a girl in a suit and tie
INFO       | 2022-11-06 17:06:29 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:29 | __main__:test_caption:46 - the digital art shows a woman with earbuds in her ears and an odd headpiece
INFO       | 2022-11-06 17:06:30 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:30 | __main__:test_caption:46 - a drawing of a girl with headphones on her ears
INFO       | 2022-11-06 17:06:31 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:31 | __main__:test_caption:46 - the lady with the yellow shirt is wearing ear phones
INFO       | 2022-11-06 17:06:32 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:32 | __main__:test_caption:46 - a woman in headphones with her hand on her hip and her eyes open
INFO       | 2022-11-06 17:06:33 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:33 | __main__:test_caption:46 - the art work shows a woman in headphones
INFO       | 2022-11-06 17:06:34 | __main__:test_caption:48 - caption: a digital painting of a woman with headphones sample: False
INFO       | 2022-11-06 17:06:34 | __main__:test_caption:50 - caption: a digital painting of a woman with headphones sample: False, num_beams=5
INFO       | 2022-11-06 17:06:35 | __main__:test_caption:52 - caption: a digital painting of a woman with headphones sample: False, num_beams=7, max_length=50
INFO       | 2022-11-06 17:06:35 | __main__:test_caption:56 - Total time: 21.93301486968994
INFO       | 2022-11-06 17:06:35 | __main__:test_caption:35 - Slow test for BLIP_Large
INFO       | 2022-11-06 17:06:36 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:36 | __main__:test_caption:46 - a woman with headphones on and a tie
INFO       | 2022-11-06 17:06:37 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:37 | __main__:test_caption:46 - a close up of a woman wearing a suit
INFO       | 2022-11-06 17:06:38 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:38 | __main__:test_caption:46 - the drawing is on paper with splatter paint on it
INFO       | 2022-11-06 17:06:39 | __main__:test_caption:45 - Num Beams: 3, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:39 | __main__:test_caption:46 - a woman wearing earphones stands looking at the camera
INFO       | 2022-11-06 17:06:39 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:39 | __main__:test_caption:46 - the woman is dressed in an unusual costume
INFO       | 2022-11-06 17:06:40 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:40 | __main__:test_caption:46 - a very colorful photo of a girl with headphones
INFO       | 2022-11-06 17:06:41 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:41 | __main__:test_caption:46 - a woman that is wearing headphones with graffiti on her face
INFO       | 2022-11-06 17:06:42 | __main__:test_caption:45 - Num Beams: 3, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:42 | __main__:test_caption:46 - a painting with people on it including men
INFO       | 2022-11-06 17:06:43 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:43 | __main__:test_caption:46 - a girl with a yellow bow in her hair
INFO       | 2022-11-06 17:06:43 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:43 | __main__:test_caption:46 - a woman with headphones on and a shirt and tie
INFO       | 2022-11-06 17:06:45 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:45 | __main__:test_caption:46 - an anime style drawing has headphones on, but it appears to be an old model
INFO       | 2022-11-06 17:06:45 | __main__:test_caption:45 - Num Beams: 3, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:45 | __main__:test_caption:46 - a drawing depicting the head and shoulders of a woman
INFO       | 2022-11-06 17:06:46 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:46 | __main__:test_caption:46 - this is an image of a woman with headphones
INFO       | 2022-11-06 17:06:47 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:47 | __main__:test_caption:46 - a very pretty young lady wearing headphones and a necklace
INFO       | 2022-11-06 17:06:48 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:48 | __main__:test_caption:46 - a close up of a cartoon girl with headphones on
INFO       | 2022-11-06 17:06:49 | __main__:test_caption:45 - Num Beams: 5, Max Length: 30, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:49 | __main__:test_caption:46 - a woman with headphones on and a watercolor painting of her face
INFO       | 2022-11-06 17:06:50 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:50 | __main__:test_caption:46 - a young lady with headphones and a suit on
INFO       | 2022-11-06 17:06:51 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:51 | __main__:test_caption:46 - the woman in the uniform has black hair
INFO       | 2022-11-06 17:06:51 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:51 | __main__:test_caption:46 - the woman is holding her headphones while staring
INFO       | 2022-11-06 17:06:53 | __main__:test_caption:45 - Num Beams: 5, Max Length: 50, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:53 | __main__:test_caption:46 - a woman with two headsets, a dress shirt, and a black tie, looks into the distance
INFO       | 2022-11-06 17:06:53 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:53 | __main__:test_caption:46 - this is a cartoon painting of a woman
INFO       | 2022-11-06 17:06:54 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.9, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:54 | __main__:test_caption:46 - a drawing of a woman in a top hat with headphones
INFO       | 2022-11-06 17:06:55 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.0
INFO       | 2022-11-06 17:06:55 | __main__:test_caption:46 - a woman in headphones and jacket holding a cellphone
INFO       | 2022-11-06 17:06:56 | __main__:test_caption:45 - Num Beams: 5, Max Length: 70, Top P: 0.95, Repetition Penalty: 1.2
INFO       | 2022-11-06 17:06:56 | __main__:test_caption:46 - girl with steampunk headphones and head scarf
INFO       | 2022-11-06 17:06:57 | __main__:test_caption:48 - caption: a drawing of a woman with headphones on sample: False
INFO       | 2022-11-06 17:06:58 | __main__:test_caption:50 - caption: a drawing of a woman with headphones on sample: False, num_beams=5
INFO       | 2022-11-06 17:06:59 | __main__:test_caption:52 - caption: a drawing of a woman with headphones on sample: False, num_beams=7, max_length=50
INFO       | 2022-11-06 17:06:59 | __main__:test_caption:56 - Total time: 23.553981065750122
```