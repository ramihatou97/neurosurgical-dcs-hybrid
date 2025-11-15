/**
 * Authentication Store
 * Pinia store for authentication state
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { authService } from '@/services/api/auth.service';
import type { User, LoginCredentials, Permission } from '@/types';

export const useAuthStore = defineStore('auth', () => {
  // State
  const user = ref<User | null>(null);
  const token = ref<string | null>(null);

  // Computed
  const isAuthenticated = computed(() => !!token.value && !!user.value);

  const userPermissions = computed(() => user.value?.permissions || []);

  // Actions
  async function login(credentials: LoginCredentials) {
    try {
      const response = await authService.login(credentials);

      token.value = response.access_token;
      user.value = response.user_info;  // Changed from response.user to match backend

      authService.storeAuth(response.access_token, response.user_info);

      return { success: true };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Login failed'
      };
    }
  }

  async function logout() {
    authService.logout();
    user.value = null;
    token.value = null;
  }

  function hasPermission(permission: Permission): boolean {
    return userPermissions.value.includes(permission);
  }

  async function initializeAuth() {
    const storedToken = authService.getStoredToken();
    const storedUser = authService.getStoredUser();

    if (storedToken && storedUser) {
      token.value = storedToken;
      user.value = storedUser;

      // Verify token is still valid by fetching current user
      try {
        const currentUser = await authService.getCurrentUser();
        user.value = currentUser;
        authService.storeAuth(storedToken, currentUser);
      } catch {
        // Token invalid, clear auth
        await logout();
      }
    }
  }

  function getStoredToken(): string | null {
    return authService.getStoredToken();
  }

  return {
    // State
    user,
    token,

    // Computed
    isAuthenticated,
    userPermissions,

    // Actions
    login,
    logout,
    hasPermission,
    initializeAuth,
    getStoredToken
  };
});
