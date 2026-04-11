<script lang="ts">
  import { dashboardContent } from './aboutContent';
</script>

<section class="section">
  <h2 class="section-title">{dashboardContent.title}</h2>
  <div class="tools-grid">
    {#each dashboardContent.features as feature}
      <div class="tool-category">
        <h3 class="tool-category-title">{feature.label}</h3>
        <p class="tool-category-desc">{feature.description}</p>
      </div>
    {/each}
  </div>
</section>
