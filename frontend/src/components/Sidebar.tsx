"use client";

import { useState } from "react";
import clsx from "clsx";

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const tabs = [
  { id: "home", label: "Home", icon: "🏠" },
  { id: "dataset", label: "Dataset", icon: "📊" },
  { id: "training", label: "Training", icon: "🧠" },
  { id: "forecast", label: "Forecast", icon: "📈" },
  { id: "insights", label: "Insights", icon: "💡" },
  { id: "reports", label: "Reports", icon: "📄" },
];

export default function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={clsx(
        "bg-slate-900 border-r border-slate-700 flex flex-col transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
    >
      <div className="p-4 border-b border-slate-700 flex items-center justify-between">
        {!collapsed && (
          <div>
            <h1 className="text-lg font-bold gradient-text">Malaria Predict</h1>
            <p className="text-xs text-slate-400">Hybrid RNN-LSTM-GRU</p>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-slate-400 hover:text-white p-1 rounded"
        >
          {collapsed ? "→" : "←"}
        </button>
      </div>

      <nav className="flex-1 py-4">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={clsx(
              "w-full flex items-center gap-3 px-4 py-3 text-sm transition-all",
              activeTab === tab.id
                ? "bg-slate-800 text-cyan-400 border-r-2 border-cyan-400"
                : "text-slate-400 hover:text-white hover:bg-slate-800/50"
            )}
            title={collapsed ? tab.label : undefined}
          >
            <span className="text-lg">{tab.icon}</span>
            {!collapsed && <span>{tab.label}</span>}
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-700">
        {!collapsed && (
          <p className="text-xs text-slate-500">
            v2.0.0 | Hybrid Deep Learning
          </p>
        )}
      </div>
    </aside>
  );
}
