import React, { useState } from 'react';
import TFTLCDDashboard from './components/TFTLCDDashboard';
import OptimizationAnalysisTools from './components/OptimizationAnalysisTools';
import './App.css';

function App() {
  const [activeView, setActiveView] = useState('dashboard');

  return (
    <div className="App">
      {/* Navigation */}
      <nav className="bg-blue-600 text-white p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold">TFT-LCD Research Dashboard</h1>
          <div className="space-x-4">
            <button
              onClick={() => setActiveView('dashboard')}
              className={`px-4 py-2 rounded ${
                activeView === 'dashboard' 
                  ? 'bg-blue-800' 
                  : 'bg-blue-500 hover:bg-blue-700'
              }`}
            >
              Data Dashboard
            </button>
            <button
              onClick={() => setActiveView('optimization')}
              className={`px-4 py-2 rounded ${
                activeView === 'optimization' 
                  ? 'bg-blue-800' 
                  : 'bg-blue-500 hover:bg-blue-700'
              }`}
            >
              Optimization Analysis
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      {activeView === 'dashboard' && <TFTLCDDashboard />}
      {activeView === 'optimization' && <OptimizationAnalysisTools />}
    </div>
  );
}

export default App;