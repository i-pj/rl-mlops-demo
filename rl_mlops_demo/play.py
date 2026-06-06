import os
import time
import warnings
import numpy as np
import gymnasium as gym
from rich.console import Console

# Suppress PyGame's internal deprecation warnings and support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", module="pkg_resources")

console = Console()

def run_manual_play() -> None:
    """Launch the CarRacing environment for manual human play."""
    try:
        import pygame
    except ImportError:
        console.print("[red]PyGame is required for manual play.[/red]")
        return

    console.print("\n[bold cyan]🎮 Starting Manual Play Mode[/bold cyan]")
    console.print("Controls:")
    console.print("  [bold]Left/Right Arrows[/bold] : Steer")
    console.print("  [bold]Up Arrow[/bold]          : Accelerate")
    console.print("  [bold]Down Arrow[/bold]        : Brake")
    console.print("\nPress [bold]ESC[/bold] or [bold]Q[/bold] to quit.\n")

    env = gym.make("CarRacing-v3", render_mode="human")
    env.reset()
    
    # action space: [steering (-1 to 1), gas (0 to 1), brake (0 to 1)]
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    total_reward = 0.0
    steps = 0
    running = True
    
    while running:
        # Process PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = -1.0
                elif event.key == pygame.K_RIGHT:
                    action[0] = +1.0
                elif event.key == pygame.K_UP:
                    action[1] = +1.0
                elif event.key == pygame.K_DOWN:
                    action[2] = +0.8
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and action[0] < 0:
                    action[0] = 0.0
                elif event.key == pygame.K_RIGHT and action[0] > 0:
                    action[0] = 0.0
                elif event.key == pygame.K_UP:
                    action[1] = 0.0
                elif event.key == pygame.K_DOWN:
                    action[2] = 0.0

        if not running:
            break

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            console.print(f"🏁 [bold green]Episode Finished![/bold green] Total Reward: [bold yellow]{total_reward:.2f}[/bold yellow] in {steps} steps.")
            
            # Contextualize against the release gate
            if total_reward >= 800:
                console.print("🌟 [bold cyan]You beat the AI model's quality release gate! (>= 800)[/bold cyan]\n")
            else:
                console.print("📉 [dim]The AI candidate requires >= 800 to pass its release gate. Better luck next time![/dim]\n")
            
            # Reset for another run
            env.reset()
            total_reward = 0.0
            steps = 0
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Gymnasium's Box2D environments usually enforce 50 FPS internally when render_mode="human",
        # but adding a small sleep prevents spinning too fast if it doesn't.
        time.sleep(0.01)

    env.close()
    console.print("[dim]Manual play exited.[/dim]")
