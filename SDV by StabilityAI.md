# SDV by StabilityAI

## Intro
- The separation of video pretraining at lower resolutions and high-quality finetuning
- Simple latent video diffu- sion baselines [8] for which we fix architecture and training scheme and assess the effect of data curation 
- Three different video training stages 
    - text-to-image pre- training 
    - video pretraining on a large dataset at low resolution 
    - high-resolution video finetuning on a much smaller dataset with higher-quality videos.

- Priors
    - Created a large Video dataset comprising 600 Million samples.
    - Trained a strong pretrained text-to-video base model that provides general motion representation.
    - Finetuned the base model on a smaller, high-quality dataset for high-resolution downstream tasks such as text- to-video.
    - Supports training using LoRA.

- Approach
    - Insert temporal convolution and attention layers after every spatial convolution and attention layer.
    - micro-conditioning on frame rate.
    - Employed the EDM-framework and significantly shift the noise schedule towards higher noise values, which we find to be essential for high-resolution finetuning.

## Data Curation and training

- Issue: Using combination of image and video data for joint image-video training that amplifies the difficulty of separating the effects of image and video on the final model.
- Solution:
        - 3 Step process:
        * Stage 1: Image pretraining ie 2D text to image diffusion model.
        * Stage 2: Video pretraining, Training on large amounts of videos.
        * Stage 3: Video finetuning, Refines model on a small subset of high quality videos at higher resolution.

- Creation:
    - Data
        - Video pretraining stage: dataset of long videos.
        - Used a cut detection pipeline in cascaded manner at 3 different FPS levels.
        - After pipeline gets 4x clips.

    - Annotation
        - Used Image captioner CoCa to annotate mid frame of each clip.
        - Used V-BLIP to obtain video based cation
        - LLM based summarisation of the first two captions.
        - Fianl result was Large Video Dataset (LVD) consisting of 580M annotated video clip pairs (212 years of content).

        - Issues:
            - Contains samples that can degrade the performance such as clips with less motion, excessive text presence, or generally low aes- thetic value
            - So additionally annotate dataset with dense optical flow at 2 FPS and removed any videos whose average optical flow magnitude is below a certain threshold.

- Training
    - Stage 1: Image Pretraining
        - Based on Stable Diffusion 2.1, trained 2 identical models on 10M subset of LVD; one with and one without pretrained spatial weights.
        - Used humans for preference in both quality and prompt-following.
    
    - Stage 2:  Curating a Video Pretraining Dataset
        - Relied on human preferences as a signal
        - Curated subsets of LVS using different methods and then considering human preference based ranking 
        - Started from randomly sampled 9.8M subset of LVD and systematically removed 12.5, 25 and 50% if examples based on CLIP scores, aesthetic scores, OCR de- tection rates, synthetic captions, optical flow scores
        - For synthetic caption filtering they used Elo ranking
        - Trained model with same hyperparams.
        - In end used Elo ranking for human preference votes.
        - After filtering, reduced dataset to LVD-10M-F and used to train baseline model that follows standard architecture and training schedule.
        - Evaluated preference scores for visual quality and prompt video alignment
        - For verification repeated same exp on a filtered subset of 50M examples and a non curated one of same size.
        - Showed more data is better.
    
    - Stage 3: High Quality Finetuning
        - Focus on inceasing resolution of training examples.
        - Used a small finetuning dataset comprising 250k pre captioned video clips of high visual fidelity.
        - Experimented with 3 identical models finetuning.
        - Model 1: Intialised with pretrained image model and skip video pretraining.
        - Model 2 and Model 3: Intialised with weights of latent video model trained on 50M curated and uncurated video clips.
        - Finetuned all for 50k steps.
        - Assess human prederence ranking early during 10k steps and at end.
        - Model 2 was best.
        - Conclusion: 
            1. The separation of video model training in video pretraining and video finetun-ing is beneficial for the final model performance after fine- tuning and that
            2. Video pretraining should ideally occur on a large scale, curated dataset, since performance differences after pretraining persist after finetuning.
    
- Training at Scale:
    - Pretrained Base Model
        - For higher resolution images, it is crucial to adopt the noise schedule when training image diffusion model, shifting towards more noise is good.
        - Firstly, Finetune the fixed discrete noise schedule from image model towards continous noise.
        - After inserting temporal layers, trained it on LVD-f on 14 frames at res 256x38.
        - Used standard EDM noise schedule for 150k iterations and 1536 batch size.
        - Finetune the model to generate 320x576 frames for 100k iterations using 768 batch size.
        - This model learnt powerful motion representation and outperforms benchmark.
    - High Res Text to Video model
        - Finetune the base text to video model on high quality video of 1M samples.
        - Consists of ject motion, steady camera motion, and well-aligned captions, and are of high visual quality.
        - finetuned on 576x1024 using 768 batch size.
    - High Res Image to Video
        - Video model receives a still input image as a conditioning and replace text embeddings with the clip embeddings.
        - Concatenated a noise augmented versionof the conditioning frame channel-wise to the input of the UNet.
        - Didnt use any masking techniques and simply copy the frame across the time axis.
        - Finetune two models, one predicting 14 frames and another one pre- dicting 25 frames.
        - They found it helpful to linearly increase the guidance scale across the frame axis (from small to high).

    - Camera Motion
        - Trained on variety of camera motion LoRAs within temporal attention blocks of model.
        - Trained on small dataset with rich camera motion metadata.
        - With 3 subsets of data categorised as "horizontally moving”, “zooming”, and “static”.
    
    - Frame Interpolation
        - For smooth videos at high frame rates, they finetuned high res text to video model into a frame interpolation model.
        - Concatenated the left and right frames to the input of the UNet via masking.
        - The model learns to predict three frames within the two conditioning frames, effectively increasing the frame rate by four.
        - Only 10k iterations were sufficed to get a good model.
    
    - Multi View Generation
        - Finetuned SVD model on 2 datasets, takes single image input and outputs a sequence of multi view images.
        - Used subset of Objaverse consisting of 150k curated and licensed synthetic 3D objects from original dataset. They rendered 360 degrees orbital videos of 21 frames with randomly sampled HDRI environment map and elevation angles (-5,30) degrees.
        - Used MVImgNet conisting of general household objects, They splited video into 200k train and 900 test videos. Also rotated frames captured in portrait mode to landscape orientation.
        - Trained model on 8 80GB A100 GPUs using batch size of 16 with lr 1e-5.
        - Named it SVD-MV
        