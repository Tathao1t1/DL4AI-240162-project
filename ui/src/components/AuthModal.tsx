/**
 * AuthModal — Sign In / Sign Up modal.
 *
 * Shown when the user is not authenticated.
 * Reuses the existing theme-input / theme-btn-primary CSS classes.
 */
import React, { useState } from 'react';
import { TrendingUp, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../contexts/AuthContext';

export const AuthModal: React.FC = () => {
  const { login, register } = useAuth();
  const [tab, setTab]       = useState<'signin' | 'signup'>('signin');
  const [email, setEmail]   = useState('');
  const [password, setPass] = useState('');
  const [confirm, setConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (tab === 'signup' && password !== confirm) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }

    setLoading(true);
    try {
      if (tab === 'signin') {
        await login(email, password);
      } else {
        await register(email, password);
      }
    } catch (err: any) {
      setError(err?.message ?? 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-md mx-4 bg-bg-surface border border-border-theme rounded-3xl shadow-2xl p-10">
        {/* Logo */}
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-accent-theme rounded-xl flex items-center justify-center shadow-lg shadow-accent-theme/20">
            <TrendingUp size={22} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tighter">TensorFinance</h1>
            <p className="text-[9px] uppercase tracking-[0.2em] text-text-muted font-bold">Financial Intelligence</p>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-1 p-1 bg-bg-deep border border-border-theme rounded-xl mb-8">
          {(['signin', 'signup'] as const).map(t => (
            <button
              key={t}
              onClick={() => { setTab(t); setError(null); }}
              className={cn(
                "flex-1 py-2 text-[11px] font-black uppercase tracking-widest rounded-lg transition-all",
                tab === t
                  ? "bg-white text-accent-theme shadow-sm border border-border-theme"
                  : "text-text-muted hover:text-text-secondary"
              )}
            >
              {t === 'signin' ? 'Sign In' : 'Sign Up'}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="theme-label">Email</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              className="theme-input w-full"
              autoComplete="email"
            />
          </div>

          <div>
            <label className="theme-label">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPass(e.target.value)}
              placeholder="••••••••"
              required
              minLength={8}
              className="theme-input w-full"
              autoComplete={tab === 'signin' ? 'current-password' : 'new-password'}
            />
          </div>

          {tab === 'signup' && (
            <div>
              <label className="theme-label">Confirm Password</label>
              <input
                type="password"
                value={confirm}
                onChange={e => setConfirm(e.target.value)}
                placeholder="••••••••"
                required
                className="theme-input w-full"
                autoComplete="new-password"
              />
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 p-3 bg-neg/10 border border-neg/20 rounded-xl text-neg text-xs font-medium">
              <AlertCircle size={15} />
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full theme-btn-primary flex items-center justify-center gap-2 mt-2"
          >
            {loading
              ? <><Loader2 size={16} className="animate-spin" /> Working…</>
              : tab === 'signin' ? 'Sign In' : 'Create Account'
            }
          </button>
        </form>
      </div>
    </div>
  );
};
