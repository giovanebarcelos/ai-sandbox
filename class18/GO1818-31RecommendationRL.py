# GO1818-31RecommendationRL
class YouTubeRecommenderEnv:
    def __init__(self, user_profile, video_catalog):
        self.user = user_profile
        self.videos = video_catalog
        self.session_watch_time = 0
        self.videos_shown = []

    def step(self, video_id):
        # Mostrar vídeo recomendado
        video = self.videos[video_id]

        # Simular comportamento usuário
        will_click = self.predict_click(self.user, video)

        if will_click:
            watch_time = self.predict_watch_time(self.user, video)
            self.session_watch_time += watch_time

            # Recompensa imediata
            reward = watch_time / 60  # Minutos assistidos

            # Bonus se usuário engajou (like, comment, share)
            engagement = self.predict_engagement(self.user, video)
            reward += engagement * 2

            # Penaliza clickbait (alto CTR mas baixo watch time)
            if watch_time < video.duration * 0.3:  # Assistiu < 30%
                reward -= 5  # Penalidade por clickbait
        else:
            # Usuário não clicou
            reward = 0

        # Usuário sai da plataforma?
        done = (len(self.videos_shown) > 20) or (np.random.random() < 0.05)

        state = self.get_user_state()
        return state, reward, done
