# PLAN: Task 11 Image Captioning

Student: M. Rifqi Dzaky Azhad  
Group: 4  
Chosen topic: Task 11, Image Captioning

## Recommended Project

**Indonesian Assistive Image Captioning for Everyday Public Scenes**

Goal: generate useful Bahasa Indonesia captions for everyday/public images, mainly for accessibility support for blind/low-vision users and public information access.

## Why This Project

1. **Impactful**: accessibility is explicitly prioritized in `context/PROPOSED_TASK.md`, and captioning helps users understand surrounding images.
2. **Not too hard**: can use existing Hugging Face/Python models and standard captioning metrics.
3. **Public relevance**: useful for public spaces, online content, education, and assistive technology.
4. **Safer than medical/disaster**: less domain risk and easier to evaluate in class.

## Verifiable Public Datasets

Use at least 2, preferably 3:

1. **Flickr30k Indonesian captions**
   - HF: `indrad123/flickr30k-transformed-captions-indonesia`
   - Useful for Indonesian caption training/evaluation.

2. **MS COCO Captions**
   - HF example: `mlgym/coco-captioning` or `phiyodr/coco2017`
   - Large standard image-caption dataset.

3. **VizWiz Captions**
   - Official: `https://vizwiz.org/tasks-and-datasets/image-captioning/`
   - Real accessibility-oriented images from blind/low-vision users.

## Models To Compare

1. **Baseline**: `nlpconnect/vit-gpt2-image-captioning`
2. **Main model**: `Salesforce/blip-image-captioning-base`
3. **Stronger model if compute allows**: `Salesforce/blip-image-captioning-large`

Optional Indonesian output strategy:

1. Fine-tune/evaluate with Indonesian Flickr30k captions.
2. Or generate English captions with BLIP, then translate to Indonesian using an open MT model.

## Experiment Plan

1. Load small dataset samples first, not full training immediately.
2. Run inference using ViT-GPT2 and BLIP.
3. Evaluate captions with BLEU, METEOR, ROUGE-L, and CIDEr if available.
4. Compare English vs Indonesian caption quality.
5. Add small human evaluation: accuracy, usefulness, and fluency.
6. Analyze common errors: wrong objects, missing context, hallucination, weak Indonesian phrasing.

## Slide Structure

1. Problem and motivation: accessibility in Indonesian context.
2. Image captioning concept: encoder-decoder / VLM.
3. Dataset explanation.
4. Model explanation.
5. Experiment setup.
6. Metrics.
7. Results table.
8. Example good captions.
9. Example bad captions.
10. Error analysis.
11. Public-use relevance.
12. Limitations.
13. Future work.
14. Individual contribution.
15. Conclusion.

## Final Decision

Choose **Indonesian Assistive Image Captioning for Everyday Public Scenes**.

Avoid healthcare, disaster satellite captioning, and surveillance for this assignment because they are more complex, need stronger domain knowledge, and are harder to validate quickly.
