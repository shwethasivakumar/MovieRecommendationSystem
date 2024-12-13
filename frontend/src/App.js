import React, { useState } from "react";
import axios from "axios";
import './MovieSearch.css'; // Ensure the updated CSS is imported

function MovieSearch() {
  const [movieName, setMovieName] = useState(""); // Movie name input state
  const [recommendations, setRecommendations] = useState([]); // Recommended movies
  const [error, setError] = useState(""); // Error message state
  const [loading, setLoading] = useState(false); // Loading state

  // Handle the movie search
  const handleSearch = async () => {
    setLoading(true); // Show loading indicator
    setError(""); // Reset error message on new search
    try {
      // Make the API call to fetch movie recommendations
      const response = await axios.get("http://127.0.0.1:5000/recommend", {
        params: {
          movie_name: movieName,
          user_id: "user1",
        },
      });

      // Debugging: Check the structure of the response
      console.log(response.data);

      const recs = response.data.recommendations; // Extract recommendations

      // If recommendations are available and are an array, update state
      if (Array.isArray(recs) && recs.length > 0) {
        setRecommendations(recs);
        setError(""); // Clear any previous errors
      } else {
        setRecommendations([]); // Clear recommendations if empty
        setError("No recommendations found. Try another movie! Example: Matilda (1996) or Monty Python and the Holy Grail (1974)");
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setRecommendations([]); // Clear recommendations on error
      setError("Error fetching recommendations.");
    }
    setLoading(false); // Hide loading indicator after completion
  };

  return (
    <div className="movie-search-container">
      <h1>Movie Recommendations</h1>
      <div className="search-box">
        <input
          type="text"
          value={movieName}
          onChange={(e) => setMovieName(e.target.value)} // Update movie name on input change
          placeholder="Enter a movie name"
          className="search-input"
        />
        <button onClick={handleSearch} className="search-button">Search</button>
      </div>

      {/* Show loading indicator while fetching */}
      {loading && <p className="loading">Loading...</p>}
      
      {/* Display error message if any */}
      {error && <p className="no-results">{error}</p>}

      {/* Show recommendations if available */}
      {Array.isArray(recommendations) && recommendations.length > 0 ? (
        <div className="recommendations">
          <ul>
            {recommendations.map((movie, index) => (
              <li key={index}>{movie}</li> // Render movie recommendations
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export default MovieSearch;
