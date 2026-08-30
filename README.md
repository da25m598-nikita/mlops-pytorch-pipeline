# mlops-pytorch-pipeline

Assignment 2: Deploying PyTorch ML Workloads with Docker & Kubernetes

A CIFAR-10 image classifier (ResNet-18) taken through the full deployment
lifecycle: local development, containerised training, and orchestrated
deployment on Kubernetes.

## Architecture

```
                    +---------------------+
                    |  configs/           |
                    |  training_config    |
                    +----------+----------+
                               |
                    mounted as ConfigMap
                               |
                               v
+-----------+        +---------------------+        +----------------+
|  CIFAR-10 | -----> |  Training Job       | -----> |  checkpoints/  |
|  dataset  |  PVC   |  (Dockerfile.train) |  PVC   |  classifier.pt |
+-----------+        +---------------------+        +--------+-------+
                                                             |
                                                     read-only PVC
                                                             |
                                                             v
                                                  +---------------------+
        client  --->  Service (80)  --------->    |  Serving Deployment |
        POST /predict                             |  (Dockerfile.serve) |
        GET  /health                              |  2 replicas, 8080   |
                                                  +---------------------+
                                                             ^
                                                             |
                                                        HPA (2-5 pods)
```

## Project structure

```
src/          model, dataset, training loop, FastAPI serving app
configs/      training hyperparameters
docker/       Dockerfile.train (multi-stage), Dockerfile.serve (non-root)
k8s/          namespace, configmap, training job, deployment, service, hpa
requirements/ pinned dependencies for training and serving
tests/        unit tests for the model
```

## Local setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements/train.txt
```

Train locally:

```
python src/train.py
```

Metrics are printed to stdout as JSON lines. The best checkpoint by
validation loss is written to `checkpoints/classifier_v1.pt`.

## Docker

Build and run the training image:

```
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1
```

Build and run the serving image:

```
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-serve:v1
```

Test the endpoints:

```
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

Quote the volume arguments if your path contains spaces.

## Kubernetes

```
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
```

Once the job completes:

```
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

Check status and test:

```
kubectl get pods -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

For a local cluster the images must be loaded in first, since they are not
pushed to a registry:

```
minikube image load mlops-train:v1
minikube image load mlops-serve:v1
```

The HPA needs metrics-server (`minikube addons enable metrics-server`).

## Notes

The requirements files pull PyTorch from the CPU-only wheel index. The
default PyPI wheels bundle NVIDIA CUDA libraries, which add roughly 2 GB to
an image that only ever runs on CPU.

The checkpoint PVC is `ReadWriteOnce`, which is fine on a single-node
cluster but would need `ReadWriteMany` for the serving Deployment to spread
across multiple nodes.

## API

`GET /health` returns 200 when a checkpoint is loaded, 503 otherwise.

`POST /predict` accepts an image as multipart form data under the key
`image` and returns the predicted class with probabilities for all ten
CIFAR-10 classes.
