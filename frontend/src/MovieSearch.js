// src/MovieSearch.js
import React, { useState } from "react";
import axios from "axios";
import "./MovieSearch.css";  // To include custom CSS for styling

const MovieSearch = () => {
  const [movieName, setMovieName] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (event) => {
    event.preventDefault();
    if (!movieName) return;

    setLoading(true);
    try {
      // Call your Flask backend to get recommendations
      const response = await axios.get(`http://localhost:5000/recommend?movie_name=${movieName}&user_id=user1`);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    }
    setLoading(false);
  };

  return (
    <div className="movie-search-container">
      <div className="search-box">
        <input
          type="text"
          placeholder="Type movie name..."
          value={movieName}
          onChange={(e) => setMovieName(e.target.value)}
          className="search-input"
        />
        <button onClick={handleSearch} className="search-button">
          Search
        </button>
      </div>

      <div className="recommendations">
        {loading ? (
          <p>Loading recommendations...</p>
        ) : (
          recommendations.length > 0 && (
            <div>
              <h3>Recommended Movies:</h3>
              <ul>
                {recommendations.map((movie, index) => (
                  <li key={index}>{movie}</li>
                ))}
              </ul>
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default MovieSearch;
