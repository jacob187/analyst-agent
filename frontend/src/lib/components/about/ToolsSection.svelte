<script lang="ts">
  import { toolsContent } from './aboutContent';
</script>

<section class="section">
  <h2 class="section-title">{toolsContent.title}</h2>

  <div class="tools-grid">
    <div class="tool-category">
      <h3 class="tool-category-title">{toolsContent.secTools.category}</h3>
      <p class="tool-category-desc">{toolsContent.secTools.description}</p>

      {#each toolsContent.secTools.tools as tool}
        <div class="tool-card">
          <div class="tool-name">{tool.name}</div>
          <div class="tool-desc">{tool.desc}</div>
        </div>
      {/each}
    </div>

    <div class="tool-category">
      <h3 class="tool-category-title">{toolsContent.yahooTools.category}</h3>
      <p class="tool-category-desc">{toolsContent.yahooTools.description}</p>

      {#each toolsContent.yahooTools.tools as tool}
        <div class="tool-card">
          <div class="tool-name">{tool.name}</div>
          <div class="tool-desc">{tool.desc}</div>
        </div>
      {/each}
    </div>
  </div>
</section>
