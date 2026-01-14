<script lang="ts">
  import { techStackContent } from './aboutContent';
</script>

<section class="section">
  <h2 class="section-title">{techStackContent.title}</h2>
  <div class="tech-grid">
    {#each techStackContent.items as item}
      <div class="tech-item">
        <div class="tech-label">{item.label}</div>
        <div class="tech-value">{item.value}</div>
      </div>
    {/each}
  </div>
</section>
