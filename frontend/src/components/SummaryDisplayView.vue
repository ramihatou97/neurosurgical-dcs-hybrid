<template>
  <div class="summary-display-view">
    <!-- Loading State -->
    <div v-if="isLoading" class="flex items-center justify-center py-12">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      <span class="ml-3 text-gray-600">Loading session data...</span>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-red-50 border border-red-200 rounded-lg p-6">
      <div class="flex items-start">
        <svg class="h-6 w-6 text-red-600 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <div class="ml-3">
          <h3 class="text-sm font-medium text-red-800">Error Loading Session</h3>
          <p class="mt-1 text-sm text-red-700">{{ error }}</p>
        </div>
      </div>
    </div>

    <!-- Session Data -->
    <div v-else-if="sessionData" class="space-y-6">
      <!-- Header with Status and Actions -->
      <div class="bg-white rounded-lg shadow-sm p-6">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-2xl font-bold text-gray-900">Discharge Summary</h2>
            <p class="text-sm text-gray-500 mt-1">Session ID: {{ sessionData.sessionId }}</p>
          </div>
          <div class="flex items-center space-x-4">
            <!-- Status Badge -->
            <span
              :class="{
                'bg-green-100 text-green-800': sessionData.status === 'completed',
                'bg-yellow-100 text-yellow-800': sessionData.status === 'pending_review',
                'bg-red-100 text-red-800': sessionData.status === 'error'
              }"
              class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium"
            >
              {{ sessionData.status }}
            </span>

            <!-- Confidence Score -->
            <div v-if="sessionData.confidenceScore" class="text-center">
              <div class="text-2xl font-bold text-primary-600">
                {{ Math.round(sessionData.confidenceScore * 100) }}%
              </div>
              <div class="text-xs text-gray-500">Confidence</div>
            </div>

            <!-- Export Button -->
            <button
              @click="exportSummary"
              class="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
            >
              Export Summary
            </button>
          </div>
        </div>

        <!-- Review Warning -->
        <div v-if="sessionData.requiresReview" class="mt-4 bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <div class="flex">
            <svg class="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
            <p class="ml-3 text-sm text-yellow-700">
              This summary requires physician review due to uncertainties detected.
            </p>
          </div>
        </div>
      </div>

      <!-- Summary Text -->
      <div class="bg-white rounded-lg shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Generated Summary</h3>
        <div class="prose prose-sm max-w-none">
          <pre class="whitespace-pre-wrap font-sans text-gray-700 leading-relaxed">{{ summaryText }}</pre>
        </div>
      </div>

      <!-- Uncertainties -->
      <div v-if="uncertainties && uncertainties.length > 0" class="bg-white rounded-lg shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">
          Uncertainties Requiring Review
          <span class="ml-2 text-sm font-normal text-gray-500">({{ uncertainties.length }})</span>
        </h3>
        <div class="space-y-3">
          <div
            v-for="(uncertainty, index) in uncertainties"
            :key="index"
            class="border rounded-lg p-4"
            :class="{
              'border-red-300 bg-red-50': uncertainty.severity === 'HIGH',
              'border-yellow-300 bg-yellow-50': uncertainty.severity === 'MEDIUM',
              'border-gray-300 bg-gray-50': uncertainty.severity === 'LOW'
            }"
          >
            <div class="flex items-start justify-between">
              <div class="flex-1">
                <div class="flex items-center">
                  <span
                    :class="{
                      'bg-red-100 text-red-800': uncertainty.severity === 'HIGH',
                      'bg-yellow-100 text-yellow-800': uncertainty.severity === 'MEDIUM',
                      'bg-gray-100 text-gray-800': uncertainty.severity === 'LOW'
                    }"
                    class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                  >
                    {{ uncertainty.severity || 'MEDIUM' }}
                  </span>
                  <span class="ml-2 text-sm font-medium text-gray-700">
                    {{ uncertainty.uncertainty_type || uncertainty.type }}
                  </span>
                </div>
                <p class="mt-2 text-sm text-gray-600">{{ uncertainty.description }}</p>
                <p v-if="uncertainty.suggested_resolution" class="mt-2 text-sm text-gray-500 italic">
                  Suggested: {{ uncertainty.suggested_resolution }}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Metrics -->
      <div v-if="metrics" class="bg-white rounded-lg shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Processing Metrics</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div class="text-center p-4 bg-gray-50 rounded-lg">
            <div class="text-2xl font-bold text-gray-900">{{ metrics.documents_processed || sessionData.documentCount }}</div>
            <div class="text-xs text-gray-500 mt-1">Documents</div>
          </div>
          <div class="text-center p-4 bg-gray-50 rounded-lg">
            <div class="text-2xl font-bold text-gray-900">{{ metrics.facts_extracted || 0 }}</div>
            <div class="text-xs text-gray-500 mt-1">Facts Extracted</div>
          </div>
          <div class="text-center p-4 bg-gray-50 rounded-lg">
            <div class="text-2xl font-bold text-gray-900">{{ formatTime(metrics.total_processing_time_ms) }}</div>
            <div class="text-xs text-gray-500 mt-1">Processing Time</div>
          </div>
          <div class="text-center p-4 bg-gray-50 rounded-lg">
            <div class="text-2xl font-bold text-gray-900">{{ Math.round((metrics.cache_hit_rate || 0) * 100) }}%</div>
            <div class="text-xs text-gray-500 mt-1">Cache Hit Rate</div>
          </div>
        </div>
      </div>

      <!-- Timeline (if available) -->
      <div v-if="timeline && timeline.key_events && timeline.key_events.length > 0" class="bg-white rounded-lg shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Clinical Timeline</h3>
        <div class="space-y-3">
          <div
            v-for="(event, index) in timeline.key_events"
            :key="index"
            class="flex items-start border-l-4 border-primary-600 pl-4 py-2"
          >
            <div class="flex-1">
              <div class="text-sm font-medium text-gray-900">{{ event.date || event.timestamp }}</div>
              <div class="text-sm text-gray-600">{{ event.event || event.description }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- No Data State -->
    <div v-else class="bg-gray-50 rounded-lg p-12 text-center">
      <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <h3 class="mt-2 text-sm font-medium text-gray-900">No session data</h3>
      <p class="mt-1 text-sm text-gray-500">Start a new session to generate a summary.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useSession } from '@/composables/useSession';

const props = defineProps<{
  sessionId: string;
}>();

const { sessionData, isLoading, error, fetchSession, exportSummary } = useSession(props.sessionId);

// Computed properties for easy access to nested data
const summaryText = computed(() => sessionData.value?.resultData?.summary_text || 'No summary available');
const uncertainties = computed(() => sessionData.value?.resultData?.uncertainties || []);
const metrics = computed(() => sessionData.value?.resultData?.metrics || null);
const timeline = computed(() => sessionData.value?.resultData?.timeline || null);

// Utility function to format milliseconds to readable time
function formatTime(ms: number | undefined): string {
  if (!ms) return '0s';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// Fetch session data on mount
onMounted(() => {
  fetchSession(props.sessionId);
});
</script>

<style scoped>
.summary-display-view {
  max-width: 1200px;
  margin: 0 auto;
}
</style>
