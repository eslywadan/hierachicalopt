// ===FILE: src/app/operation_model/websocket/realtime-updates.ts===

import { Server as HTTPServer } from 'http';
import { Server as SocketIOServer, Socket } from 'socket.io';
import { EventEmitter } from 'events';

/**
 * WebSocket server for real-time updates
 */
export class RealtimeUpdateServer extends EventEmitter {
  private io: SocketIOServer;
  private clients: Map<string, Socket> = new Map();
  
  constructor(httpServer: HTTPServer) {
    super();
    this.io = new SocketIOServer(httpServer, {
      cors: {
        origin: '*',
        methods: ['GET', 'POST']
      }
    });
    
    this.setupConnectionHandlers();
  }
  
  private setupConnectionHandlers(): void {
    this.io.on('connection', (socket: Socket) => {
      console.log(`Client connected: ${socket.id}`);
      this.clients.set(socket.id, socket);
      
      // Handle client events
      socket.on('subscribe', (channel: string) => {
        socket.join(channel);
        console.log(`Client ${socket.id} subscribed to ${channel}`);
      });
      
      socket.on('unsubscribe', (channel: string) => {
        socket.leave(channel);
        console.log(`Client ${socket.id} unsubscribed from ${channel}`);
      });
      
      socket.on('disconnect', () => {
        console.log(`Client disconnected: ${socket.id}`);
        this.clients.delete(socket.id);
      });
      
      // Level 3 specific events
      socket.on('requestPrediction', (data) => {
        this.emit('predictionRequest', { socketId: socket.id, data });
      });
      
      socket.on('requestValidation', (data) => {
        this.emit('validationRequest', { socketId: socket.id, data });
      });
    });
  }
  
  /**
   * Broadcast training progress to all subscribers
   */
  broadcastTrainingProgress(progress: any): void {
    this.io.to('training').emit('trainingProgress', progress);
  }
  
  /**
   * Send prediction results to specific client
   */
  sendPredictionResults(socketId: string, results: any): void {
    const socket = this.clients.get(socketId);
    if (socket) {
      socket.emit('predictionResults', results);
    }
  }
  
  /**
   * Broadcast model status updates
   */
  broadcastModelStatus(status: any): void {
    this.io.emit('modelStatus', status);
  }
  
  /**
   * Broadcast Little's Law violations
   */
  broadcastViolation(violation: {
    timestamp: Date;
    plant: string;
    deviation: number;
    message: string;
  }): void {
    this.io.to('monitoring').emit('littlesLawViolation', violation);
  }
  
  /**
   * Send performance metrics
   */
  broadcastMetrics(metrics: any): void {
    this.io.to('metrics').emit('performanceMetrics', metrics);
  }
}
