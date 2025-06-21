import React, { createContext, useContext, useEffect, useState } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);            // { email }  or  null

  // put auth state in localStorage so it survives refresh
  useEffect(() => {
    const saved = localStorage.getItem('pp-user');
    if (saved) setUser(JSON.parse(saved));
  }, []);

  const login = (email, password) => {
    // hard-coded demo credentials
    if (email === 'demo@presentperfect.ai' && password === '1w?H[M!?0F2M') {
      const u = { email };
      setUser(u);
      localStorage.setItem('pp-user', JSON.stringify(u));
      return true;
    }
    return false;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('pp-user');
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);