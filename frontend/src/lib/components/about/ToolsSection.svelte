<script lang="ts">
  import { capabilitiesContent } from './aboutContent';
</script>

<section class="section">
  <h2 class="section-title">{capabilitiesContent.title}</h2>

  <div class="tools-grid">
    {#each capabilitiesContent.categories as cat}
      <div class="tool-category">
        <h3 class="tool-category-title">{cat.category}</h3>
        <p class="tool-category-desc">{cat.description}</p>
      </div>
    {/each}
  </div>
</section>
