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


if __name__ == '__main__':
    import numpy as np
    np.random.seed(7)

    print("=== Simulação de Recomendação RL (YouTube-style) ===")

    # Perfil e catálogo mínimos
    class User:
        interests = ["tech", "gaming"]

    class Video:
        def __init__(self, vid_id, title, duration, category):
            self.id = vid_id
            self.title = title
            self.duration = duration
            self.category = category

    user_profile = User()
    catalog = {
        0: Video(0, "Python Tutorial",    600, "tech"),
        1: Video(1, "Cat Compilations",   300, "pets"),
        2: Video(2, "Game Review 2025",   900, "gaming"),
        3: Video(3, "Cooking Pasta",      480, "food"),
        4: Video(4, "Deep Learning Intro",1200, "tech"),
    }

    env = YouTubeRecommenderEnv(user_profile, catalog)

    # Métodos de simulação inline (necessários para o step)
    env.predict_click = lambda u, v: v.category in u.interests
    env.predict_watch_time = lambda u, v: v.duration * 0.8 if v.category in u.interests else v.duration * 0.1
    env.predict_engagement = lambda u, v: 1 if v.category in u.interests else 0
    env.get_user_state = lambda: [len(env.videos_shown), env.session_watch_time / 60]

    print(f"  Sessão de recomendação:")
    total_reward = 0
    for t in range(5):
        video_id = t % len(catalog)
        state, reward, done = env.step(video_id)
        video = catalog[video_id]
        clicou = video.category in user_profile.interests
        print(f"    Vídeo: '{video.title}' "
              f"| clicou={'Sim' if clicou else 'Não'} | reward={reward:+.2f}")
        total_reward += reward
        if done:
            break

    print(f"\n  Tempo assistido total: {env.session_watch_time:.0f}s "
          f"({env.session_watch_time / 60:.1f} min)")
    print(f"  Recompensa total: {total_reward:.2f}")
