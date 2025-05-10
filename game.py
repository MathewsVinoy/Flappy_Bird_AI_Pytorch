import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Pygame Window")

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Game loop
def main():
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Fill the screen with white
        screen.fill(WHITE)
        
        # Update the display
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(60)

    # Quit Pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()