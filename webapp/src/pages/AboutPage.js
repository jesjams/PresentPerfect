import React from 'react';

import kaviImg from '../assets/Kavi.jpg';
import saloniImg from '../assets/Saloni.jpg';
import jonathanImg from '../assets/Jonathan.jpg';
import adityaImg from '../assets/aditya.jpg';

export default function AboutPage() {
  const styles = {
    page: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#f9f9ff',
      padding: '2rem',
      overflow: 'hidden',
      boxSizing: 'border-box',
    },
    sectionTitle: {
      color: '#5D2E8C',
      fontSize: '2rem',
      textAlign: 'center',
      marginBottom: '.5rem',
    },
    description: {
      maxWidth: '800px',
      margin: '0 auto 2 rem',
      color: '#333',
      fontSize: '1rem',
      lineHeight: '1.2',
      textAlign: 'center',
    },
    teamGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem',
      maxWidth: '1300px',
      width: '100%',
    },
    card: {
      backgroundColor: '#ffffff',
      borderRadius: '12px',
      boxShadow: '0 8px 20px rgba(0,0,0,0.15)',
      padding: '2rem',
      textAlign: 'center',
      transition: 'transform 0.2s ease',
    },
    avatar: {
      width: '180px',
      height: '180px',
      borderRadius: '50%',
      objectFit: 'cover',
      marginBottom: '1rem',
    },
    name: {
      fontSize: '1.3rem',
      fontWeight: 600,
      marginBottom: '.5rem',
      color: '#333',
    },
    role: {
      fontSize: '1rem',
      fontWeight: 500,
      marginBottom: '1rem',
      color: '#5D2E8C',
    },
    bio: {
      fontSize: '1rem',
      color: '#555',
      lineHeight: '1',
    },
    triangleBg: {
      position: 'fixed',
      top: '40%',
      right: 0,
      width: '100%',
      height: '60%',
      opacity: '10%',
      backgroundColor: '#B288C0',
      zIndex: -1,
      clipPath: 'polygon(70% 15%, 0% 100%, 85% 100%)',
      pointerEvents: 'none'
    },
  };

const team = [
  { img: kaviImg, name: 'Kavi Sathyamurthy', role: 'Developer', bio: 'More than half my friends think I am a Vampire' },
  { img: saloniImg, name: 'Saloni Samant', role: 'Developer', bio: 'Neutralizes the 0.01% germs sanitizers cannot' },
  { img: jonathanImg, name: 'Jonathan Sjamsudin', role: 'Developer', bio: 'Professional Yapper, Part-Time Data Analyst' },
  { img: adityaImg, name: 'Aditya Maniar', role: 'Developer', bio: 'my brain is less “gray matter” and more “internet soup.”' },
];

  return (
    <>
      <div className="about-page" style={styles.page}>
        <h1 className="section-title" style={styles.sectionTitle}>About Present Perfect</h1>
        <p style={styles.description}>
          Present Perfect is a smart presentation-coaching web app that analyzes your delivery,
          gaze, gestures, and speech in real time. It uses advanced NLP and computer vision
          models to give you actionable feedback—helping you become a more confident and
          engaging speaker.
        </p>

        <h2 className="section-title" style={{ ...styles.sectionTitle, fontSize: '3rem', marginTop: '0rem' }}>
          Meet the Team
        </h2>
        <div className="team-grid" style={styles.teamGrid}>
          {team.map(({ img, name, role, bio }) => (
            <div
              key={name}
              className="card"
              style={styles.card}
              onMouseEnter={e => (e.currentTarget.style.transform = 'scale(1.05)')}
              onMouseLeave={e => (e.currentTarget.style.transform = 'scale(1)')}
            >
              <img src={img} alt={name} style={styles.avatar} />
              <div style={styles.name}>{name}</div>
              <div style={styles.role}>{role}</div>
              <div style={styles.bio}>{bio}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={styles.triangleBg} />

      <style>{`
        body, html { margin: 0; padding: 0; overflow: auto; }
        @media (min-width: 1200px) {
          body, html { overflow: hidden; }
        }
        @media (max-width: 1200px) {
          .about-page { display: block !important; padding: 2rem !important; height: auto !important; overflow: auto !important; }
          .section-title { font-size: 2.5rem !important; margin-bottom: 1.5rem !important; }
          .team-grid { display: grid !important; grid-template-columns: 1fr !important; gap: 3rem !important; }
          .card { padding: 2rem !important; }
          .card img { width: 140px !important; height: 140px !important; margin-bottom: 1rem !important; }
        }
      `}</style>
    </>
  );
}