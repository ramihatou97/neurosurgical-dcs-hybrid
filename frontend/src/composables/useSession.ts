/**
 * Session Composable
 * Handles fetching and managing processing session data
 */

import { ref, Ref } from 'vue';
import { processingService } from '@/services/api/processing.service';

interface SessionData {
  sessionId: string;
  status: string;
  createdAt: string;
  completedAt?: string;
  documentCount: number;
  confidenceScore?: number;
  requiresReview: boolean;
  resultData: any;
}

export function useSession(sessionId?: Ref<string | null> | string | null) {
  const sessionData = ref<SessionData | null>(null);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  /**
   * Fetch session data from API
   */
  async function fetchSession(id?: string): Promise<void> {
    const targetId = id || (typeof sessionId === 'string' ? sessionId : sessionId?.value);

    if (!targetId) {
      error.value = 'No session ID provided';
      return;
    }

    isLoading.value = true;
    error.value = null;

    try {
      const data = await processingService.getSession(targetId);
      sessionData.value = data;
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to fetch session';
      console.error('Session fetch error:', err);
      sessionData.value = null;
    } finally {
      isLoading.value = false;
    }
  }

  /**
   * Export summary as text
   */
  function exportSummary(): void {
    if (!sessionData.value?.resultData?.summary_text) {
      error.value = 'No summary available to export';
      return;
    }

    const summary = sessionData.value.resultData.summary_text;
    const blob = new Blob([summary], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `discharge-summary-${sessionData.value.sessionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return {
    sessionData,
    isLoading,
    error,
    fetchSession,
    exportSummary
  };
}
