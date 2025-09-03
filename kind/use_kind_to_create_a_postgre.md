## Running PostgreSQL on kind with kubectl

1. **Edit deployment YAML files:**
- In this yaml, it create a pv (persistent volumem), pvc (persistent volume claim), container and service.
```yaml
   ---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: postgres-pv
spec:
  capacity:
    storage: 3Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /Users/qs.chou/projects/hierachicalopt/kind/pv/postgres # <-- Change this to your desired host path
  storageClassName: standard
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
    - ReadWriteOnce # This access mode means the volume can be mounted as read-write by a single node.
  resources:
    requests:
      storage: 3Gi # Request 5GB of storage. Adjust as needed.
  volumeName: postgres-pv
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-deployment
  labels:
    app: postgres
spec:
  replicas: 1 # Number of PostgreSQL instances. For production, consider StatefulSets for highly available setups.
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:13 # Use a specific PostgreSQL image version.
          env:
            - name: POSTGRES_DB
              value: mydatabase # Name of the database to create.
            - name: POSTGRES_USER
              value: myuser # Username for the database.
            - name: POSTGRES_PASSWORD
              value: mypassword # Password for the database. **Use Kubernetes Secrets in production.**
          ports:
            - containerPort: 5432 # Default PostgreSQL port.
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data # Mount point for PostgreSQL data.
      volumes:
        - name: postgres-storage
          persistentVolumeClaim:
            claimName: postgres-pvc # Link to the PersistentVolumeClaim.
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
spec:
  selector:
    app: postgres
  ports:
    - protocol: TCP
      port: 5432 # Service port.
      targetPort: 5432 # Port on the pod.
 
   ```


2. **Apply the deployment files to your kind cluster:**

   ```sh
   kubectl apply -f ./kind/deployment/postgresql-deployment.yaml
   ```

3. **Verify the pods and service:**

   ```sh
   kubectl get pods -n postgres
   kubectl get svc -n postgres-service
   ```

   4. **Preload the image to speed up the pod creation**
- preload the image 
```sh
podman pull postgres:15-alpine
```
- 
```sh
   kind load docker-image <image-name>
```