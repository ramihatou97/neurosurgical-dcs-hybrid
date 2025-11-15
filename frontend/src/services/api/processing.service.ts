/**
 * Processing Service
 * Handles discharge summary generation
 */

import { apiClient } from './client';
import type { ProcessingRequest, ProcessingResult, UncertaintyResolution } from '@/types/processing.types';

export class ProcessingService {
  /**
   * Process documents to generate discharge summary
   *
   * @param request - Processing request with documents
   * @returns Processing result with session ID
   */
  async processDocuments(request: ProcessingRequest): Promise<ProcessingResult> {
    const response = await apiClient.post<ProcessingResult>(
      '/api/process',
      {
        documents: request.documents,
        options: request.options || {},
        use_parallel: request.useParallel ?? true,
        use_cache: request.useCache ?? true,
        apply_learning: request.applyLearning ?? true
      }
    );

    return response;
  }

  /**
   * Get processing session by ID
   *
   * @param sessionId - Processing session ID
   * @returns Complete session data including result
   */
  async getSession(sessionId: string): Promise<any> {
    return apiClient.get<any>(`/api/sessions/${sessionId}`);
  }

  /**
   * Submit uncertainty resolutions
   *
   * @param sessionId - Processing session ID
   * @param resolutions - Uncertainty resolutions
   */
  async resolveUncertainties(
    sessionId: string,
    resolutions: UncertaintyResolution[]
  ): Promise<void> {
    await apiClient.post(`/api/process/${sessionId}/resolve`, {
      resolutions
    });
  }
}

export const processingService = new ProcessingService();
