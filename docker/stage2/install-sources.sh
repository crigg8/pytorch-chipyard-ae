#!/usr/bin/env bash
set -euo pipefail

root=/opt/chipyard
support=/opt/stage2-support
while read -r destination repository commit archive_sha; do
  [[ -z "$destination" || "$destination" == \#* ]] && continue
  mkdir -p "$root/$destination"
  git -C "$root/$destination" init -q
  git -C "$root/$destination" remote add origin "$repository"
  if [[ -n "$archive_sha" ]]; then
    # Large Linux Git transfers can stall while GitHub generates a pack.
    # Import the checksummed source archive into a local Git repository so
    # FireMarshal can still track modifications. This import commit is not
    # the upstream revision; sources.lock records the upstream identity.
    github_path="${repository#https://github.com/}"
    curl -fL --retry 5 "https://codeload.github.com/${github_path%.git}/tar.gz/$commit" -o /tmp/stage2-source.tar.gz
    printf '%s  %s\n' "$archive_sha" /tmp/stage2-source.tar.gz | sha256sum -c -
    tar xzf /tmp/stage2-source.tar.gz --no-same-owner --strip-components=1 -C "$root/$destination"
    rm /tmp/stage2-source.tar.gz
    git -C "$root/$destination" add -f --all
    GIT_AUTHOR_DATE=2000-01-01T00:00:00Z GIT_COMMITTER_DATE=2000-01-01T00:00:00Z \
      git -C "$root/$destination" -c user.name='Stage2 source import' -c user.email=source@localhost \
      commit -q -m "Import $repository at $commit (archive SHA-256 $archive_sha)"
  else
    git -C "$root/$destination" fetch --depth 1 origin "$commit"
    git -C "$root/$destination" checkout -q --detach FETCH_HEAD
    test "$(git -C "$root/$destination" rev-parse HEAD)" = "$commit"
  fi
done < "$support/sources.lock"

git -C "$root/sims/firesim" apply "$support/patches/firesim.patch"
git -C "$root/software/firemarshal" apply "$support/patches/firemarshal.patch"
git -C "$root/software/firemarshal/boards/default/linux" apply "$support/patches/linux-rvv.patch"

curl -fL --retry 5 https://deb.debian.org/debian/pool/main/f/fakeroot/fakeroot_1.31.orig.tar.gz \
  -o /tmp/fakeroot.tar.gz
printf '%s\n' '63886d41e11c56c7170b9d9331cca086421b350d257338ef14daad98f77e202f  /tmp/fakeroot.tar.gz' | sha256sum -c -
tar xzf /tmp/fakeroot.tar.gz --no-same-owner -C "$root/software/firemarshal/boards/default/distros/br"
rm /tmp/fakeroot.tar.gz

cat > "$root/env.sh" <<'EOF'
export CHIPYARD_DIR=/opt/chipyard
export CONDA_PREFIX=/opt/chipyard/.conda-env
export RISCV=/opt/chipyard/.conda-env/riscv-tools
export PATH="$CONDA_PREFIX/bin:$RISCV/bin:$PATH"
EOF
cat > "$root/sims/firesim/env.sh" <<'EOF'
source /opt/chipyard/env.sh
export FIRESIM_ENV_SOURCED=1
export FS_DIR=/opt/chipyard/sims/firesim
EOF
