BUCKET_NAME=BUCKET_NAME
JOB_NAME=JOB_NAME
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models
REGION=us-central1
IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-gpu.1-7

cat > config.yaml <<EOF
trainingInput:
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: epoch_val_acc
    maxTrials: 30
    maxParallelTrials: 2
    enableTrialEarlyStopping: True
    params:
    - parameterName: learning-rate
      type: DOUBLE
      minValue: 0.0001
      maxValue: 1
      scaleType: UNIT_LOG_SCALE
EOF

# Submit to GCP
gcloud ai-platform jobs submit training ${JOB_NAME} \
 --package-path trainer/ \
 --module-name trainer.train \
 --master-image-uri ${IMAGE_URI} \
 --region ${REGION} \
 --scale-tier BASIC_GPU \
 --job-dir ${JOB_DIR} \
 --config config.yaml \
 -- \
 --num-epochs 25 \
 --model-name "unet_model.pth" \
 --net "UNet" \
 --lr-scheduler "OneCycleLR" \
 --batch-size 42

# Stream the logs from the job
gcloud ai-platform jobs stream-logs $JOB_NAME