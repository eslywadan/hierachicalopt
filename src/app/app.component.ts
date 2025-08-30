import { Component } from '@angular/core';
import { RouterOutlet, RouterLink } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink],
  template: `
    <div class="app-container">
      <header class="app-header">
        <h1>Hierarchical Optimization</h1>
        <nav>
          <a routerLink="/opt-board" class="nav-link">Opt Board</a>
          <a routerLink="/operation-model" class="nav-link">Operation Model</a>
        </nav>
      </header>
      <main class="main-content">
        <router-outlet></router-outlet>
      </main>
    </div>
  `,
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'hierachicalopt';
}