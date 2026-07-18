# frozen_string_literal: true
#
# This repo is both an Obsidian vault and a Jekyll site. The Markdown sources use
# file-relative asset paths (e.g. ../assets/foo.png) so images and attachments
# resolve in Obsidian and in GitHub's preview. For the built website those same
# targets must be site-absolute (/assets/foo.png), because pages live under
# /posts/:title/ and /teaching/:name/ where a relative ../assets/ would 404.
#
# This hook rewrites the target of any Markdown link or image that points under
# assets/ to a site-absolute path, at build time. It is anchored on the "](" of
# the link syntax, so ordinary prose is untouched, and it is idempotent.
#
# Handled forms:  ](assets/x)  ](/assets/x)  ](../assets/x)  ](../../assets/x)
# Result:         ](/assets/x)

module AssetsPathRewrite
  # `](`  optional spaces  zero-or-more `../`  optional leading `/`  `assets/`
  PATTERN = %r{\]\(\s*(?:\.\./)*/?assets/}

  def self.rewrite(content)
    content.gsub(PATTERN, "](/assets/")
  end
end

Jekyll::Hooks.register %i[posts pages documents], :pre_render do |doc|
  content = doc.content
  doc.content = AssetsPathRewrite.rewrite(content) if content
end
