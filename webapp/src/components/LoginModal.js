import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';

export default function LoginModal({ open, onClose, title = 'Welcome Back' }) {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [pass, setPass] = useState('');
  const [error, setError] = useState('');

  if (!open) return null;

  const tryLogin = e => {
    e.preventDefault();
    if (login(email, pass)) {
      setError('');
      onClose();
    } else {
      setError('Invalid credentials. Use demo@presentperfect.ai / 1w?H[M!?0F2M');
    }
  };

  const backdrop = {
    position: 'fixed', top: 0, left: 0,
    width: '100vw', height: '100vh',
    background: 'rgba(0,0,0,0.5)',
    backdropFilter: 'blur(4px)',
    display: 'flex', alignItems: 'center',
    justifyContent: 'center', zIndex: 3000
  };

  const card = {
    background: '#fff',
    borderRadius: '1rem',
    padding: '2rem',
    width: '90%',
    maxWidth: '400px',
    position: 'relative',
    boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
    fontFamily: 'sans-serif',
    boxSizing: 'border-box'
  };

  const closeBtn = {
    position: 'absolute',
    top: '0.5rem',
    right: '0.5rem',
    background: 'none',
    border: 'none',
    fontSize: '1.2rem',
    cursor: 'pointer',
    color: '#888'
  };

  const label = { display: 'block', marginBottom: '0.25rem', fontWeight: 500, fontSize: '0.9rem' };

  const input = {
    width: '100%',
    padding: '0.6rem',
    marginBottom: '1rem',
    borderRadius: '0.5rem',
    border: '1px solid #ccc',
    fontSize: '0.95rem',
    outline: 'none',
    transition: 'border-color 0.2s',
    boxSizing: 'border-box'
  };

  const button = {
    width: '100%',
    padding: '0.75rem',
    border: 'none',
    borderRadius: '0.6rem',
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
    backgroundColor: 'rgba(93, 46, 140, 0.84)',
    color: '#fff',
    boxSizing: 'border-box',
    transition: 'opacity 0.2s'
  };

  return (
    <div style={backdrop} onClick={onClose}>
      <form style={card} onClick={e => e.stopPropagation()} onSubmit={tryLogin}>
        <button style={closeBtn} onClick={onClose} aria-label="Close">×</button>

        <h2 style={{ marginTop: 0, marginBottom: '1rem', textAlign: 'center' }}>{title}</h2>

        <label style={label}>Email</label>
        <input
          type="email"
          placeholder="you@example.com"
          value={email}
          onChange={e => setEmail(e.target.value)}
          style={input}
          onFocus={e => (e.target.style.borderColor = '#8f5dd1')}
          onBlur={e => (e.target.style.borderColor = '#ccc')}
          required
        />

        <label style={label}>Password</label>
        <input
          type="password"
          placeholder="••••••••"
          value={pass}
          onChange={e => setPass(e.target.value)}
          style={input}
          onFocus={e => (e.target.style.borderColor = '#8f5dd1')}
          onBlur={e => (e.target.style.borderColor = '#ccc')}
          required
        />

        {error && (
          <p style={{ color: 'crimson', marginBottom: '1rem', textAlign: 'center' }}>{error}</p>
        )}

        <button
          type="submit"
          style={button}
          onMouseEnter={e => (e.currentTarget.style.opacity = '0.9')}
          onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
        >
          Log In
        </button>
      </form>
    </div>
  );
}
