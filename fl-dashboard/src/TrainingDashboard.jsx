// src/TrainingDashboard.jsx
import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import "./TrainingDashboard.css";

const TrainingDashboard = () => {
  const [status, setStatus] = useState({
    round: 0,
    loss: 0.0,
    accuracy: 0.0,
    aggregation_time: 0.0,
    avg_encryption_time: 0.0,
    avg_decryption_time: 0.0,
    log: [],
  });
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(
          "http://localhost:5000/status?timestamp=" + new Date().getTime(),
          { cache: "no-store" }
        );
        const data = await response.json();
        console.log("Fetched training status:", data);
        setStatus(data);
        if (data.round > 0) {
          setHistory((prevHistory) => {
            if (prevHistory.find((d) => d.round === data.round)) return prevHistory;
            return [
              ...prevHistory,
              { round: data.round, loss: data.loss, accuracy: data.accuracy },
            ];
          });
        }
      } catch (error) {
        console.error("Error fetching training status:", error);
      }
    };

    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Federated Learning Dashboard (Plain)</h1>
        <p>Hoàng Hải Anh - 21521819 | Trương Khánh Long - 21521750</p>
      </header>
      <div className="dashboard-content">
        <div className="left-panel">
          <div className="metrics-cards">
            <div className="card metric-card">
              <h2>Round</h2>
              <p>{status.round}</p>
            </div>
            <div className="card metric-card">
              <h2>Loss</h2>
              <p>{status.loss.toFixed(6)}</p>
            </div>
            <div className="card metric-card">
              <h2>Accuracy</h2>
              <p>{(status.accuracy * 100).toFixed(2)}%</p>
            </div>
            <div className="card metric-card">
              <h2>Aggregation Time (s)</h2>
              <p>{status.aggregation_time ? status.aggregation_time.toFixed(4) : "N/A"}</p>
            </div>
            <div className="card metric-card">
              <h2>Avg Encryption Time (s)</h2>
              <p>{status.avg_encryption_time ? status.avg_encryption_time.toFixed(4) : "N/A"}</p>
            </div>
            <div className="card metric-card">
              <h2>Avg Decryption Time (s)</h2>
              <p>{status.avg_decryption_time ? status.avg_decryption_time.toFixed(4) : "N/A"}</p>
            </div>
          </div>
          <div className="logs-section">
            <h2>Training Logs</h2>
            <ul className="logs-list">
              {status.log.map((entry, index) => (
                <li key={index}>{entry}</li>
              ))}
            </ul>
          </div>
        </div>
        <div className="right-panel">
          <div className="chart-container">
            <h2>Training Progress</h2>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={history} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: "Round", position: "insideBottomRight", offset: -5 }} />
                <YAxis
                  yAxisId="left"
                  label={{ value: "Loss", angle: -90, position: "insideLeft" }}
                  domain={[0, 1]}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  label={{ value: "Accuracy", angle: -90, position: "insideRight" }}
                  domain={[0, 1]}
                />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#8884d8" activeDot={{ r: 8 }} name="Loss" />
                <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#82ca9d" name="Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingDashboard;
