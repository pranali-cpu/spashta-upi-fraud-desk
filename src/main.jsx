import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
window.SPASHTA_API_URL = import.meta.env.VITE_API_URL || "";
createRoot(document.getElementById("root")).render(<App />);
