/**
 * AuthContext — global auth state (user + token).
 *
 * On mount: attempts to restore session from localStorage token.
 * Provides: user, loading, login, register, logout.
 */
import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import {
  AuthUser,
  login as svcLogin,
  register as svcRegister,
  logout as svcLogout,
  fetchMe,
  getToken,
} from '../services/authService';

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  login:    (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout:   () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser]       = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  // Restore session on mount
  useEffect(() => {
    if (!getToken()) { setLoading(false); return; }
    fetchMe()
      .then(setUser)
      .catch(() => {/* expired token — stay logged out */})
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const u = await svcLogin(email, password);
    setUser(u);
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    await svcRegister(email, password);
    // Auto-login after registration
    const u = await svcLogin(email, password);
    setUser(u);
  }, []);

  const logout = useCallback(() => {
    svcLogout();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}
