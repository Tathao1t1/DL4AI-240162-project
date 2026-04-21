/**
 * authService.ts — JWT auth calls to the FastAPI backend.
 *
 * Token is stored in localStorage under key 'qp_token'.
 */

const BASE = '/api/v1/auth';
const TOKEN_KEY = 'qp_token';

export interface AuthUser {
  id: string;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  is_verified: boolean;
}

/** Read stored token (null if not logged in). */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

function saveToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

export function authHeader(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/** Register a new account. Throws on failure. */
export async function register(email: string, password: string): Promise<AuthUser> {
  const res = await fetch(`${BASE}/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail ?? `Register failed (${res.status})`);
  }
  return res.json();
}

/** Log in, store token, return user. Throws on failure. */
export async function login(email: string, password: string): Promise<AuthUser> {
  // fastapi-users login expects application/x-www-form-urlencoded
  const form = new URLSearchParams({ username: email, password });
  const res = await fetch(`${BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form.toString(),
  });
  if (!res.ok) {
    throw new Error('Invalid email or password.');
  }
  const data = await res.json();
  saveToken(data.access_token);
  return fetchMe();
}

/** Fetch current user info (requires stored token). */
export async function fetchMe(): Promise<AuthUser> {
  const res = await fetch(`${BASE}/me`, {
    headers: authHeader(),
  });
  if (!res.ok) throw new Error('Not authenticated');
  return res.json();
}

/** Remove stored token (client-side logout). */
export function logout() {
  clearToken();
}
