# Install 
1. Install kind 
2. use kind to create cluster
3. install kubectl 
## On Mac M1
- install kind
```bash
brew install kind
```

- Config kind to use podman rather than default Docker
```bash
export KIND_EXPERIMENTAL_PROVIDER=podman
kind create cluster
```