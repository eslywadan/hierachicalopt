# Setup Instructions for Level 3 LSTM Operation Model

## Prerequisites

Ensure you have the following installed:
- Node.js (v18 or higher)
- npm or yarn
- Angular CLI (`npm install -g @angular/cli`)

## Installation Steps

### 1. Install Required Dependencies

Navigate to your project root and run:

```bash
npm install @tensorflow/tfjs chart.js sequelize sequelize-typescript socket.io socket.io-client
npm install @angular/material @angular/cdk
npm install --save-dev @types/node
```

### 2. Update Angular Configuration

Add the following to your `tsconfig.json`:

```json
{
  "compilerOptions": {
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true
  }
}
```

### 3. Add Material Theme

In `src/styles.scss`, add:

```scss
@import '@angular/material/prebuilt-themes/indigo-pink.css';

html, body { height: 100%; }
body { margin: 0; font-family: Roboto, "Helvetica Neue", sans-serif; }
```

### 4. File Structure Setup

Create the following directory structure in `src/app/operation_model/`:

```
src/app/operation_model/
├── models/
│   └── level3-lstm-model.ts
├── services/
│   └── level3-operation.service.ts
├── components/
│   ├── level3-dashboard/
│   │   ├── level3-dashboard.component.ts
│   │   ├── level3-dashboard.component.html
│   │   └── level3-dashboard.component.scss
│   └── level3-chart/
│       └── level3-chart.component.ts
├── utils/
│   ├── data-utils.ts
│   └── littles-law-calculator.ts
├── websocket/
│   └── realtime-updates.ts
├── database/
│   └── models.ts
├── __tests__/
│   └── level3-lstm-model.test.ts
├── operation-model.module.ts
└── README.md
```

### 5. Copy the Code Files

Copy all the provided code files into their respective locations as shown above.

### 6. Update App Routing

Update `src/app/app.routes.ts` to include the operation model route:

```typescript
export const routes: Routes = [
  // ... existing routes
  {
    path: 'operation-model',
    loadChildren: () => import('./operation_model/operation-model.module').then(m => m.OperationModelModule)
  }
];
```

### 7. Add Navigation (Optional)

If you have a navigation component, add a link to the Level 3 model:

```html
<a routerLink="/operation-model">Level 3 Operation Model</a>
```

## Running the Application

### Development Server

```bash
ng serve
```

Navigate to `http://localhost:4200/operation-model`

### First Time Setup

1. Click **"Initialize Model"** to set up the model architecture
2. Click **"Train Model"** to generate training data and train the LSTM
3. Wait for training to complete (progress bar will show status)
4. Once trained, you can make predictions

### Making Predictions

1. Select a **Plant**, **Application**, and **Panel Size**
2. Enter the number of **Prediction Days** (1-30)
3. Set **Current WIP** and **Planned Throughput**
4. Optionally set a **Target Production** goal
5. Click **"Generate Predictions"**

## Testing

### Run Unit Tests

```bash
ng test
```

### Run Specific Tests

```bash
ng test --include='**/*level3*.spec.ts'
```

## Troubleshooting

### Common Issues and Solutions

#### 1. TensorFlow.js Not Loading

If you see errors about TensorFlow not being found:

```bash
npm install @tensorflow/tfjs@latest --force
```

#### 2. Material Components Not Rendering

Ensure Material modules are imported in `operation-model.module.ts`

#### 3. Chart.js Issues

If charts aren't displaying:

```bash
npm install chart.js@latest
```

#### 4. Memory Issues During Training

Reduce batch size or sequence length in the configuration:

```typescript
const lstmConfig = {
  batchSize: 16,  // Reduce from 32
  sequenceLength: 7,  // Reduce from 14
  // ... other config
};
```

## Performance Optimization

### 1. Use Web Workers for Training

For better performance, consider moving LSTM training to a Web Worker:

```typescript
// Create worker file: src/app/operation_model/workers/training.worker.ts
addEventListener('message', async ({ data }) => {
  // Training logic here
  postMessage({ type: 'progress', value: progress });
});
```

### 2. Enable GPU Acceleration

TensorFlow.js can use WebGL for GPU acceleration:

```typescript
import * as tf from '@tensorflow/tfjs';

// Check if WebGL is available
console.log('WebGL backend available:', tf.env().get('WEBGL_VERSION'));
```

### 3. Implement Caching

Cache trained models in IndexedDB:

```typescript
// Save model
await model.save('indexeddb://level3-lstm-model');

// Load model
const model = await tf.loadLayersModel('indexeddb://level3-lstm-model');
```

## Production Build

### Build for Production

```bash
ng build --configuration production
```

### Optimize Bundle Size

Add to `angular.json`:

```json
{
  "optimization": {
    "scripts": true,
    "styles": true,
    "fonts": true
  },
  "budgets": [
    {
      "type": "bundle",
      "name": "main",
      "maximumWarning": "2mb",
      "maximumError": "5mb"
    }
  ]
}
```

## Integration with Backend API

### Node.js Express Server Example

```typescript
// server.js
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

// Training data endpoint
app.get('/api/training-data', async (req, res) => {
  // Fetch from database
  const data = await TrainingData.findAll();
  res.json(data);
});

// Prediction endpoint
app.post('/api/predict', async (req, res) => {
  const { plant, application, panelSize, days } = req.body;
  // Process prediction
  const result = await predictProduction(req.body);
  res.json(result);
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
```

### Update Service to Use API

```typescript
// In level3-operation.service.ts
getTrainingData(): Observable<DataPoint[]> {
  return this.http.get<DataPoint[]>('/api/training-data');
}

makePredictionAPI(request: PredictionRequest): Observable<any> {
  return this.http.post('/api/predict', request);
}
```

## Monitoring and Logging

### Add Application Insights

```typescript
import { ApplicationInsights } from '@microsoft/applicationinsights-web';

const appInsights = new ApplicationInsights({
  config: {
    instrumentationKey: 'YOUR_KEY',
    enableAutoRouteTracking: true
  }
});

appInsights.loadAppInsights();
appInsights.trackEvent({ name: 'ModelTrained', properties: { accuracy: 0.95 } });
```

## Next Steps

1. **Customize Data Generation**: Modify the `DataGenerator` class to use real historical data
2. **Enhance Model Architecture**: Experiment with different LSTM configurations
3. **Add More Features**: Include additional input features like quality metrics, machine availability
4. **Implement Real-time Updates**: Set up WebSocket connections for live predictions
5. **Create API Documentation**: Document your REST API endpoints using Swagger/OpenAPI

## Support

For issues or questions, check:
- Angular Documentation: https://angular.io/docs
- TensorFlow.js Documentation: https://www.tensorflow.org/js
- Material Design: https://material.angular.io