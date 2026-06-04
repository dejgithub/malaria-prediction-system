import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Malaria Prediction System | Hybrid RNN-LSTM-GRU",
  description: "Production-ready Malaria Disease Prediction System using Hybrid Deep Learning Model",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
