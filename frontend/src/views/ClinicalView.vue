<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <div class="bg-white shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold text-gray-900">
              Discharge Summary Generator
            </h1>
            <p class="mt-1 text-sm text-gray-600">
              Medical-grade discharge summary generation with human verification
            </p>
          </div>

          <!-- User info -->
          <div class="flex items-center space-x-4">
            <div class="text-right">
              <div class="text-sm font-medium text-gray-900">
                {{ user?.full_name || user?.username }}
              </div>
              <div class="text-xs text-gray-500">
                {{ user?.role }}
              </div>
            </div>

            <button
              @click="handleLogout"
              class="text-sm text-gray-600 hover:text-gray-900"
            >
              Logout
            </button>
          </div>
        </div>

        <!-- Progress Indicator -->
        <div class="mt-6">
          <div class="flex items-center justify-between">
            <!-- Step 1 -->
            <div class="flex items-center flex-1">
              <div
                class="flex items-center justify-center w-10 h-10 rounded-full text-sm font-medium"
                :class="currentStep >= 1 ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-600'"
              >
                1
              </div>
              <div class="ml-3">
                <div class="text-sm font-medium text-gray-900">Import</div>
                <div class="text-xs text-gray-500">Add documents</div>
              </div>
            </div>

            <!-- Connector -->
            <div class="flex-1 h-1 mx-4" :class="currentStep >= 2 ? 'bg-primary-600' : 'bg-gray-200'"></div>

            <!-- Step 2 -->
            <div class="flex items-center flex-1">
              <div
                class="flex items-center justify-center w-10 h-10 rounded-full text-sm font-medium"
                :class="currentStep >= 2 ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-600'"
              >
                2
              </div>
              <div class="ml-3">
                <div class="text-sm font-medium text-gray-900">Verify</div>
                <div class="text-xs text-gray-500">Review & confirm</div>
              </div>
            </div>

            <!-- Connector -->
            <div class="flex-1 h-1 mx-4" :class="currentStep >= 3 ? 'bg-primary-600' : 'bg-gray-200'"></div>

            <!-- Step 3 -->
            <div class="flex items-center flex-1">
              <div
                class="flex items-center justify-center w-10 h-10 rounded-full text-sm font-medium"
                :class="currentStep >= 3 ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-600'"
              >
                3
              </div>
              <div class="ml-3">
                <div class="text-sm font-medium text-gray-900">Generate</div>
                <div class="text-xs text-gray-500">Create summary</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Error Display -->
      <div
        v-if="error"
        class="mb-6 bg-red-50 border border-red-200 rounded-lg p-4"
      >
        <div class="flex">
          <svg class="h-5 w-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
          </svg>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">Error</h3>
            <p class="mt-1 text-sm text-red-700">{{ error }}</p>
          </div>
        </div>
      </div>

      <!-- Warnings Display -->
      <div
        v-if="hasWarnings && currentStep === 2"
        class="mb-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4"
      >
        <div class="flex">
          <svg class="h-5 w-5 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
          </svg>
          <div class="ml-3 flex-1">
            <h3 class="text-sm font-medium text-yellow-800">
              Parser Warnings ({{ parseWarnings.length }})
            </h3>
            <ul class="mt-2 text-sm text-yellow-700 list-disc list-inside space-y-1">
              <li v-for="(warning, idx) in parseWarnings" :key="idx">
                {{ warning }}
              </li>
            </ul>
            <p class="mt-2 text-xs text-yellow-600">
              Please review the documents below carefully and make any necessary corrections.
            </p>
          </div>
        </div>
      </div>

      <!-- Step 1: Document Input -->
      <div v-if="currentStep === 1" class="bg-white rounded-lg shadow-sm p-8">
        <!-- Input Method Tabs -->
        <div class="mb-8 border-b border-gray-200">
          <nav class="-mb-px flex space-x-8">
            <button
              @click="inputMethod = 'bulk'"
              :class="[
                'py-4 px-1 border-b-2 font-medium text-sm',
                inputMethod === 'bulk'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              ]"
            >
              Bulk Import
              <span class="ml-2 text-xs bg-primary-100 text-primary-800 px-2 py-0.5 rounded-full">
                Recommended
              </span>
            </button>

            <button
              @click="inputMethod = 'individual'"
              :class="[
                'py-4 px-1 border-b-2 font-medium text-sm',
                inputMethod === 'individual'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              ]"
            >
              Individual Upload
              <span class="ml-2 text-xs bg-gray-200 text-gray-600 px-2 py-0.5 rounded-full">
                Coming Soon
              </span>
            </button>
          </nav>
        </div>

        <!-- Bulk Import -->
        <div v-if="inputMethod === 'bulk'">
          <BulkTextInput
            v-model:bulkText="bulkText"
            v-model:separatorType="separatorType"
            v-model:customSeparator="customSeparator"
            :isLoading="isLoading"
            :validationError="error"
            @parse="handleParse"
            @reset="handleReset"
          />
        </div>

        <!-- Individual Upload (Placeholder) -->
        <div v-else class="text-center py-12">
          <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <h3 class="mt-2 text-sm font-medium text-gray-900">Individual Upload</h3>
          <p class="mt-1 text-sm text-gray-500">
            Upload documents one at a time. Coming soon.
          </p>
        </div>
      </div>

      <!-- Step 2: Review & Verify -->
      <div v-if="currentStep === 2" class="space-y-6">
        <!-- Header with stats -->
        <div class="bg-white rounded-lg shadow-sm p-6">
          <div class="flex items-center justify-between">
            <div>
              <h2 class="text-2xl font-bold text-gray-900">
                Step 2: Review & Verify Documents
              </h2>
              <p class="mt-1 text-sm text-gray-600">
                Please verify each document's type and date before proceeding.
                <span class="font-medium text-red-600">This step is mandatory.</span>
              </p>
            </div>

            <div class="text-right">
              <div class="text-3xl font-bold text-gray-900">
                {{ verifiedDocs.length }}
              </div>
              <div class="text-sm text-gray-500">
                Documents to verify
              </div>
            </div>
          </div>

          <!-- Low confidence warning -->
          <div
            v-if="lowConfidenceCount > 0"
            class="mt-4 bg-yellow-50 border border-yellow-200 rounded-md p-3"
          >
            <p class="text-sm text-yellow-800">
              <span class="font-medium">{{ lowConfidenceCount }}</span>
              {{ lowConfidenceCount === 1 ? 'document has' : 'documents have' }}
              low confidence scores. Please review carefully.
            </p>
          </div>
        </div>

        <!-- Document Cards -->
        <div class="space-y-4">
          <DocumentReviewCard
            v-for="doc in verifiedDocs"
            :key="doc.index"
            :document="doc"
            @update="(updates) => updateVerifiedDocument(doc.index, updates)"
            @delete="deleteDocument(doc.index)"
          />
        </div>

        <!-- Action Buttons -->
        <div class="bg-white rounded-lg shadow-sm p-6">
          <div class="flex items-center justify-between">
            <button
              @click="handleBackToStep1"
              class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
            >
              ← Back to Import
            </button>

            <div class="flex items-center space-x-4">
              <div class="text-sm text-gray-600">
                <span v-if="allDocumentsVerified" class="text-green-700 font-medium">
                  ✓ All documents verified
                </span>
                <span v-else class="text-yellow-700 font-medium">
                  Please verify all documents
                </span>
              </div>

              <button
                @click="handleProcessDocuments"
                :disabled="!allDocumentsVerified || isLoading"
                class="px-6 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span v-if="isLoading" class="flex items-center">
                  <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </span>
                <span v-else>
                  Generate Summary →
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 3: Summary Display -->
      <div v-if="currentStep === 3">
        <SummaryDisplayView v-if="sessionId" :session-id="sessionId" />

        <!-- Start New Session Button -->
        <div class="mt-6 text-center">
          <button
            @click="handleStartOver"
            class="px-6 py-2 text-sm font-medium text-primary-600 bg-primary-50 border border-primary-200 rounded-md hover:bg-primary-100"
          >
            Start New Session
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useMultiNoteProcessor } from '@/composables/useMultiNoteProcessor';
import { useAuth } from '@/composables/useAuth';
import { useToast } from '@/composables/useToast';
import BulkTextInput from '@/components/clinical/BulkTextInput.vue';
import DocumentReviewCard from '@/components/clinical/DocumentReviewCard.vue';
import SummaryDisplayView from '@/components/SummaryDisplayView.vue';

/**
 * ClinicalView - Main Clinical Workflow
 *
 * THREE-STEP PROGRESSIVE DISCLOSURE:
 * 1. Document Input (Bulk or Individual)
 * 2. Review & Verify (MANDATORY Human Gate)
 * 3. Summary Generation & Review
 *
 * SAFETY ARCHITECTURE:
 * - Step 2 is mandatory and cannot be bypassed
 * - All documents must be human-verified before processing
 * - Processing calls existing /api/process endpoint (187 tests)
 */

const router = useRouter();
const { user, logout } = useAuth();
const { success, error: showError } = useToast();

// Multi-note processor state
const {
  bulkText,
  separatorType,
  customSeparator,
  suggestedDocs,
  verifiedDocs,
  parseWarnings,
  isLoading,
  error,
  hasWarnings,
  lowConfidenceCount,
  allDocumentsVerified,
  parseBulkText,
  updateVerifiedDocument,
  deleteDocument,
  processVerifiedDocuments,
  reset
} = useMultiNoteProcessor();

// UI state
const inputMethod = ref<'bulk' | 'individual'>('bulk');
const currentStep = ref<1 | 2 | 3>(1);
const sessionId = ref<string | null>(null);

// Computed
const canProceedToStep2 = computed(() => verifiedDocs.value.length > 0);

// Handlers
async function handleParse(): Promise<void> {
  const success = await parseBulkText();

  if (success) {
    currentStep.value = 2;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

function handleBackToStep1(): void {
  currentStep.value = 1;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function handleProcessDocuments(): Promise<void> {
  const result = await processVerifiedDocuments();

  if (result) {
    sessionId.value = result.sessionId;
    currentStep.value = 3;
    success('Documents processed successfully!');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

function handleReset(): void {
  if (confirm('Are you sure you want to clear all data and start over?')) {
    reset();
    currentStep.value = 1;
    sessionId.value = null;
  }
}

function handleStartOver(): void {
  reset();
  currentStep.value = 1;
  sessionId.value = null;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function handleLogout(): Promise<void> {
  await logout();
  router.push('/login');
}
</script>
