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
  const demoAccounts = [
    { email: 'demo@presentperfect.ai', password: '1w?H[M!?0F2M' },
    { email: 'kavi@presentperfect.ai', password: '1212123q' },
    { email: 'jonathan@presentperfect.ai', password: '1212123q' },
  ];

  const matchedUser = demoAccounts.find(
    (acc) => acc.email === email && acc.password === password
  );

  if (matchedUser) {
    const u = { email: matchedUser.email };
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