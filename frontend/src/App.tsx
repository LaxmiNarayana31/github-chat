import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import LandingPage from './components/LandingPage';
import ChatPage from './components/ChatPage';

const STORAGE_KEY = "github-chat-conversations";

const App: React.FC = () => {
  // Clear localStorage on app start (when server restarts/page loads fresh)
  useEffect(() => {
    // Check if this is a fresh page load (not a navigation)
    const isFirstLoad = sessionStorage.getItem("app-loaded") !== "true";
    if (isFirstLoad) {
      localStorage.removeItem(STORAGE_KEY);
      sessionStorage.setItem("app-loaded", "true");
      console.log("🧹 LocalStorage cleared on app start");
    }
  }, []);

  return (
    <BrowserRouter>
      <Toaster position="top-center" />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/chat" element={<ChatPage />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;