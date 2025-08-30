
// ===FILE: src/app/operation_model/utils/littles-law-calculator.ts===

/**
 * Little's Law Calculator and Validator
 */
export class LittlesLawCalculator {
  /**
   * Calculate cycle time using Little's Law
   * CT = WIP / Throughput
   */
  static calculateCycleTime(wip: number, throughput: number): number {
    if (throughput === 0) {
      throw new Error('Throughput cannot be zero');
    }
    return wip / throughput;
  }
  
  /**
   * Calculate WIP using Little's Law
   * WIP = Throughput × CT
   */
  static calculateWIP(throughput: number, cycleTime: number): number {
    return throughput * cycleTime;
  }
  
  /**
   * Calculate throughput using Little's Law
   * Throughput = WIP / CT
   */
  static calculateThroughput(wip: number, cycleTime: number): number {
    if (cycleTime === 0) {
      throw new Error('Cycle time cannot be zero');
    }
    return wip / cycleTime;
  }
  
  /**
   * Validate if values comply with Little's Law
   */
  static validate(
    wip: number,
    throughput: number,
    cycleTime: number,
    tolerance: number = 0.05
  ): {
    isValid: boolean;
    expectedCycleTime: number;
    actualCycleTime: number;
    deviation: number;
    deviationPercentage: number;
  } {
    const expectedCycleTime = this.calculateCycleTime(wip, throughput);
    const deviation = Math.abs(expectedCycleTime - cycleTime);
    const deviationPercentage = (deviation / expectedCycleTime) * 100;
    const isValid = deviationPercentage <= tolerance * 100;
    
    return {
      isValid,
      expectedCycleTime,
      actualCycleTime: cycleTime,
      deviation,
      deviationPercentage
    };
  }
  
  /**
   * Calculate required changes to achieve target
   */
  static calculateRequiredChanges(
    current: { wip: number; throughput: number; cycleTime: number },
    target: { wip?: number; throughput?: number; cycleTime?: number }
  ): {
    requiredWIPChange?: number;
    requiredThroughputChange?: number;
    requiredCycleTimeChange?: number;
    recommendations: string[];
  } {
    const result: any = {};
    const recommendations: string[] = [];
    
    if (target.cycleTime !== undefined && target.cycleTime !== current.cycleTime) {
      // Calculate required changes to achieve target cycle time
      if (target.wip !== undefined) {
        // WIP is fixed, calculate required throughput
        const requiredThroughput = target.wip / target.cycleTime;
        result.requiredThroughputChange = requiredThroughput - current.throughput;
        recommendations.push(
          `Adjust throughput to ${requiredThroughput.toFixed(2)} units/day`
        );
      } else if (target.throughput !== undefined) {
        // Throughput is fixed, calculate required WIP
        const requiredWIP = target.throughput * target.cycleTime;
        result.requiredWIPChange = requiredWIP - current.wip;
        recommendations.push(
          `Adjust WIP to ${requiredWIP.toFixed(0)} units`
        );
      } else {
        // Both can change, provide options
        const option1WIP = current.throughput * target.cycleTime;
        const option2Throughput = current.wip / target.cycleTime;
        
        recommendations.push(
          `Option 1: Keep throughput at ${current.throughput}, adjust WIP to ${option1WIP.toFixed(0)}`,
          `Option 2: Keep WIP at ${current.wip}, adjust throughput to ${option2Throughput.toFixed(2)}`
        );
      }
    }
    
    return { ...result, recommendations };
  }
  
  /**
   * Calculate system efficiency
   */
  static calculateEfficiency(
    actualThroughput: number,
    theoreticalThroughput: number
  ): number {
    if (theoreticalThroughput === 0) return 0;
    return (actualThroughput / theoreticalThroughput) * 100;
  }
}

// ===FILE: src/app/operation_model/services/level3-operation.service.ts===