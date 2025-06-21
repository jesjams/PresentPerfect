import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import logo from '../assets/logo.png';
import LoginModal from './LoginModal';
import { useAuth } from '../context/AuthContext';

export default function Header() {
  const location = useLocation();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { user, logout } = useAuth();
  const [showLogin, setShowLogin] = useState(false);

  const navItems = [
    { path: '/upload', label: 'Home' },
    { path: '/about', label: 'About' },
  ];

  const isActive = path => location.pathname === path;

  const baseLink = {
    padding: '0.4rem 1rem',
    borderRadius: '1rem',
    border: '1px solid transparent',
    textDecoration: 'none',
    fontWeight: 500,
    transition: 'all 0.2s ease-in-out',
    cursor: 'pointer',
    display: 'inline-block',
  };

  const styles = {
    header: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '1rem 2rem',
      borderBottom: '2px solid #5D2E8C',
      backgroundColor: '#f9f9ff',
      position: 'relative',
      zIndex: 1000,
    },
    leftGroup: { display: 'flex', alignItems: 'center', gap: '2rem' },
    logoContainer: { display: 'flex', alignItems: 'center', height: '30px' },
    logoImage: { height: '160%', objectFit: 'contain' },
    nav: { display: 'flex', gap: '1rem', alignItems: 'start', marginTop: '10px', fontSize: '16px' },
    rightGroup: { display: 'flex', alignItems: 'center', gap: '1rem' },
    drawerButton: { fontSize: '1.5rem', background: 'none', border: 'none', cursor: 'pointer' },
    overlay: {
      position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
      backgroundColor: 'rgba(0,0,0,0.3)', display: drawerOpen ? 'block' : 'none', zIndex: 1500,
    },
    drawer: {
      position: 'fixed', top: 0, left: 0, height: '100%', width: '250px',
      backgroundColor: '#f9f9ff', boxShadow: 'rgba(0, 0, 0, 0.3) 4px 0px 8px', padding: '2rem',
      display: 'flex', flexDirection: 'column', gap: '1rem',
      transform: drawerOpen ? 'translateX(0)' : 'translateX(-100%)',
      transition: 'transform 0.3s ease-in-out', zIndex: 2000,
    },
  };

  const getLinkStyle = active => ({
    ...baseLink,
    backgroundColor: active ? '#5D2E8C' : 'transparent',
    color: active ? '#f9f9ff' : 'black',
    borderColor: active ? 'black' : 'transparent',
  });

  return (
    <>
      <header className="header" style={styles.header}>
        <div style={styles.leftGroup}>
          <div style={styles.logoContainer}>
            <img src={logo} alt="Present Perfect Logo" style={styles.logoImage} />
          </div>
          <nav style={styles.nav} className="nav">
            {navItems.map(({ path, label }) => (
              <Link
                key={path}
                to={path}
                className="nav-link"
                style={getLinkStyle(isActive(path))}
              >
                {label}
              </Link>
            ))}
          </nav>
        </div>

        <div style={styles.rightGroup}>
          {user ? (
            <button
              onClick={logout}
              className="auth-button nav-link"
              style={getLinkStyle(false)}
              title="Click to logout"
            >
              {user.email}
            </button>
          ) : (
            <button
              onClick={() => setShowLogin(true)}
              className="auth-button nav-link"
              style={getLinkStyle(false)}
            >
              Login
            </button>
          )}

          <button
            className="drawer-button"
            style={styles.drawerButton}
            onClick={() => setDrawerOpen(o => !o)}
          >
            ☰
          </button>
        </div>
      </header>

      <div style={styles.overlay} onClick={() => setDrawerOpen(false)} />

      <div style={styles.drawer} className="drawer">
        {navItems.map(({ path, label }) => (
          <Link
            key={path}
            to={path}
            className="nav-link"
            style={getLinkStyle(isActive(path))}
            onClick={() => setDrawerOpen(false)}
          >
            {label}
          </Link>
        ))}

        {/* Auth in drawer */}
        {user ? (
          <button
            onClick={() => { logout(); setDrawerOpen(false); }}
            className="auth-button"
            style={getLinkStyle(false)}
          >
            Logout
          </button>
        ) : (
          <button
            onClick={() => { setShowLogin(true); setDrawerOpen(false); }}
            className="auth-button"
            style={getLinkStyle(false)}
          >
            Login
          </button>
        )}
      </div>

      {/* Login modal */}
      <LoginModal open={showLogin} onClose={() => setShowLogin(false)} />

      <style>
        {`
          /* Hide hamburger by default */
          .drawer-button { display: none; }

          /* Responsive toggles */
          @media (max-width: 768px) {
            .nav { display: none !important; }
            .header .auth-button { display: none !important; }
            .drawer-button { display: block !important; }
            .drawer .auth-button { display: block !important; }
          }

          /* Ensure consistent box-sizing */
          .nav-link, .auth-button { box-sizing: border-box; }

          /* Sidebar link styling */
          .drawer .nav-link {
            display: block;
            width: 100%;
            text-align: left;
            border: none !important;
          }

          /* Default login button background */
          .auth-button {
            background-color: #5D2E8C !important;
            color: #f9f9ff !important;
          }
          .auth-button:hover {
            background-color: rgba(93, 46, 140, 0.84) !important;
          }

          /* Hover and active feedback */
          .nav-link:hover, .auth-button:hover {
            background-color: rgba(93, 46, 140, 0.84) !important;
            color: #f9f9ff !important;
          }
          .nav-link:active, .auth-button:active {
            filter: brightness(85%) !important;
          }
        `}
      </style>
    </>
  );
}
