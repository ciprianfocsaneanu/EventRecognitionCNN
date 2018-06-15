set BUCKET_NAME=licenta-datasets
set JOB_NAME="train_$(date +%Y%m%d_%H%M%S)"
set JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
set REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.0 \
  --module-name trainer.train \
  --package-path ./trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --data_directory gs://$BUCKET_NAME/subset27x2000


  gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/$JOB_NAME --runtime-version 1.0 --module-name trainer.train --package-path ./trainer --region $REGION --config=trainer/cloudml-gpu.yaml -- --data_directory gs://$BUCKET_NAME/subset27x2000

  gcloud ml-engine jobs submit training job3 --job-dir gs://licenta-datasets/job3 --runtime-version 1.0 --module-name trainer.train --package-path ./trainer --region us-east1 --config=trainer/cloudml_gpu.yaml -- --data_directory gs://licenta-datasets/places205-subset27

  
#### New version

### Train locally using gcloud

`gcloud ml-engine local train --module-name trainer.train --package-path ./trainer --job-dir ./jobs`


### Train on Google Cloud ML Engine

gcloud ml-engine jobs submit training job0_12 --job-dir gs://licenta-storage/jobs-events/job0_12 --module-name cloud-trainer.train --package-path ./cloud-trainer --region us-east1 --config=cloud-trainer/cloudml_gpu.yaml -- --data_bucket gs://licenta-storage/ --data_file wider-dataset.h5 --context_model_file densenet121-model.h5

export BUCKET_NAME=licenta-datasets
export JOB_NAME="train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.0 \
  --module-name trainer.train \
  --package-path ./trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --data_directory gs://$BUCKET_NAME/subset27x2000



