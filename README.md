
# Hit Song Prediction Using Spotify Data

In 2025, music industry analysts noticed something surprising: for the first time since 1990, not a single hip-hop song appeared in the US Billboard Top 40. This sparked a lot of debate online about whether hip-hop was “falling off,” changing direction, or simply shifting toward different listening habits. That discussion inspired this project. Instead of arguing based on opinions, I wanted to explore the question with data: what actually makes a song a hit? Using Spotify’s audio features, metadata, and artist information, this project builds a machine-learning system that predicts whether a song is likely to become a hit based purely on its characteristics. Although the motivation comes from hip-hop trends, the final dataset includes songs from all genres so the model can learn general hit-song patterns rather than genre-specific quirks.

The project starts with a basic dataset of songs containing titles, artists, and Spotify audio features such as danceability, energy, tempo, acousticness, and valence. I wrote a script (spotify_query.py) that connects to the Spotify Web API and enriches each track with additional attributes like release date, album type, explicit flag, Spotify popularity, artist popularity, and genre tags. This step produces a fully enhanced dataset called spotify_enriched.csv, which provides the metadata needed to study hit-song patterns.

The machine-learning workflow (hit_song.py) follows standard supervised-learning practices. I define a “hit” as any song in the top 25% of track popularity on Spotify—similar to how charts measure relative success. Then I preprocess the data using a scikit-learn ColumnTransformer, which scales numeric features and applies one-hot encoding to categorical ones. I train three models: a DummyClassifier baseline, Logistic Regression, and an SVM with an RBF kernel. After cross-validation and hyperparameter tuning, the SVM performs the best, achieving test accuracy around 87% and showing slightly stronger separation in its ROC curve compared with Logistic Regression. These ROC curves, included in the figures folder, illustrate how both models provide strong ranking ability, with the SVM maintaining a small but consistent edge.

To interpret what drives success, I examine logistic regression coefficients and permutation-based feature importance for the SVM. The permutation importance plot shows that artist popularity dominates all other predictors, emphasizing the real-world advantage that large, established fanbases have on streaming behavior. Beyond that, features such as valence, release year, energy, loudness, instrumentalness, and various key signatures contribute meaningfully to the model. These findings align with the logistic-regression coefficient rankings and highlight consistent patterns across linear and nonlinear models. I also visualize how individual audio features differ between hits and non-hits using violin plots for danceability, energy, tempo, and valence. These plots reveal clear distribution shifts—hits tend to be slightly more energetic, brighter in mood, and marginally more danceable—providing an intuitive, human-interpretable link to the model’s learned behavior.

Overall, this project uses real audio data and modern machine-learning techniques to analyze what contributes to commercial music success. It approaches the “hip-hop falloff” conversation from a data-driven perspective and provides a reproducible framework for studying music trends, predicting hits, and understanding how artistic and acoustic features interact with listeners’ preferences at scale.

To run the project end-to-end, users must first supply their own Spotify API credentials. The script spotify_query.py requires a personal Spotify client ID and client secret, which should be set as environment variables (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET) before running any data collection. Once the keys are configured, running spotify_query.py pulls audio features, metadata, and artist information from Spotify and writes the combined output to spotify_enriched.csv. After the dataset is created, the full modeling pipeline can be executed by running hit_song.py, which trains the models, prints all evaluation results directly to the terminal, and saves the generated charts—including ROC curves, permutation-importance plots, and violin distributions—into the figures/ directory. This provides a complete and reproducible workflow, with all results visible either in the terminal or in the generated images.


Final SVM test accuracy: 87%

Baseline accuracy: ~74.6%

Most important feature: artist popularity



Future Improvements:
- Try Random Forest or XGBoost
- Test different definitions of a "hit"
- Add more recent Spotify data