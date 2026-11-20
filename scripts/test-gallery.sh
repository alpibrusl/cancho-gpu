#!/bin/sh
# A gallery for testing the studio's groups (docs/studio-sessions.md, section 20): pictures for each group, with the group each should have.
#
#     scripts/test-gallery.sh generate DIR URL TOKEN          queue one text-to-image job per line of studio/testgallery.tsv in a running studio (about 50 s each on this machine;
#                                                              clean examples made from the prompts of the list), and write DIR/jobs.tsv (FROM=n starts at line n and appends)
#     scripts/test-gallery.sh collect DIR DATA                after the jobs are done: copy each image.png out of the studio's data directory DATA into DIR as NN-group.png
#     scripts/test-gallery.sh fetch DIR                       or: download real photographs from Wikimedia Commons (the first result of each search; NOT clean examples:
#                                                              a search for a wool sweater returned a handwritten letter) into DIR as NN-group.jpg, with credits.tsv
#     scripts/test-gallery.sh load DIR URL TOKEN [blind]      add the pictures of DIR to a running studio, in the list's order; the group is set as the list says,
#                                                              unless `blind`: then no group and no description are given, and the describer sorts and describes them
#     scripts/test-gallery.sh score DIR DATA                  after a blind load into an EMPTY data directory DATA and the describer's passes: how many pictures have the group the list says
#
# Nothing is committed (the pictures go where you say). Needs curl and jq.
set -eu
UA="cancho-gpu-tests/1.0 (studio test gallery; a few requests, slowly)"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LIST=$ROOT/studio/testgallery.tsv
# the picture of line $1 of the list (group $2) in DIR: NN-group.png (generated) or NN-group.jpg (fetched), or nothing
pic() { f=$(printf '%s/%02d-%s' "$DIR" "$1" "$2"); if [ -s "$f.png" ]; then echo "$f.png"; elif [ -s "$f.jpg" ]; then echo "$f.jpg"; fi; }
# the prompt of a picture: what the list says it shows, in the way that group is photographed
prompt_of() {
  case "$1" in
    person) echo "A photograph of $2, the whole person in view, natural light";;
    garment) echo "A product photograph of $2 on a plain white background, no person";;
    furniture) echo "A product photograph of $2 in a plain studio, no people";;
    room) echo "A photograph of $2, a wide view of the whole room, no people";;
    place) echo "A photograph of $2, a wide view, natural light";;
    object) echo "A photograph of $2 on a plain table, no people";;
    *) echo "A sharp photograph of $2 in natural light";;
  esac
}
cmd=${1:-}
case "$cmd" in
generate)
  DIR=${2:?usage: test-gallery.sh generate DIR URL TOKEN}; URL=${3:?}; TOKEN=${4:?}
  FROM=${FROM:-1}
  mkdir -p "$DIR"; [ "$FROM" = 1 ] && : > "$DIR/jobs.tsv"
  n=0
  while IFS=$(printf '\t') read -r group query desc; do
    n=$((n + 1))
    [ "$n" -ge "$FROM" ] || continue
    P=$(prompt_of "$group" "$desc")
    R=$(curl -sS -H "Authorization: Bearer $TOKEN" -XPOST -H 'Content-Type: application/json' "$URL/api/jobs" -d "$(jq -n --arg p "$P" --argjson s "$n" '{kind:"image",prompt:$p,size:"768x768",seed:$s}')")
    id=$(printf '%s' "$R" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "refused: $R" >&2; exit 1; }
    printf '%02d\t%s\t%s\n' "$n" "$group" "$id" >> "$DIR/jobs.tsv"
  done < "$LIST"
  echo "queued the jobs of the lines $FROM to $n: $DIR/jobs.tsv"
  ;;
collect)
  DIR=${2:?usage: test-gallery.sh collect DIR DATA}; DATA=${3:?}
  while IFS=$(printf '\t') read -r n group id; do
    # a line that was queued again later replaces the earlier one
    if [ -s "$DATA/out/$id/image.png" ]; then cp "$DATA/out/$id/image.png" "$DIR/$n-$group.png"; fi
  done < "$DIR/jobs.tsv"
  got=0; missing=0; i=0
  while IFS=$(printf '\t') read -r group query desc; do
    i=$((i + 1))
    if [ -n "$(pic "$i" "$group")" ]; then got=$((got + 1)); else missing=$((missing + 1)); echo "no picture yet: line $i ($group)" >&2; fi
  done < "$LIST"
  echo "collected: $got pictures, $missing missing (queue the missing again with FROM=<line>)"
  ;;
fetch)
  DIR=${2:?usage: test-gallery.sh fetch DIR}
  mkdir -p "$DIR"
  : > "$DIR/credits.tsv"
  n=0
  while IFS=$(printf '\t') read -r group query desc; do
    n=$((n + 1))
    f=$(printf '%s/%02d-%s.jpg' "$DIR" "$n" "$group")
    if [ -s "$f" ]; then continue; fi
    q=$(printf '%s' "$query" | sed 's/ /%20/g')
    ok=0; try=0
    while [ "$ok" = 0 ] && [ "$try" -lt 3 ]; do
      try=$((try + 1))
      J=$(curl -sS -m 60 -A "$UA" "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=filetype:bitmap%20$q&gsrnamespace=6&gsrlimit=12&prop=imageinfo&iiprop=url%7Cextmetadata%7Csize&iiurlwidth=640&format=json" 2>/dev/null || true)
      row=$(printf '%s' "$J" | jq -r '[.query.pages[]? | select(.imageinfo[0].width >= 500 and .imageinfo[0].height >= 400) | select((.imageinfo[0].extmetadata.LicenseShortName.value // "") | test("^(CC0|Public domain|CC BY)")) | select(.imageinfo[0].thumburl != null)] | sort_by(.index)[0] | if . == null then empty else [.imageinfo[0].thumburl, ((.imageinfo[0].extmetadata.Artist.value // "unknown") | gsub("<[^>]*>"; "") | gsub("[\t\n]"; " ")), .imageinfo[0].extmetadata.LicenseShortName.value, ("https://commons.wikimedia.org/?curid=" + (.pageid | tostring))] | @tsv end' 2>/dev/null || true)
      if [ -n "$row" ]; then
        url=$(printf '%s' "$row" | cut -f1)
        sleep 2
        if curl -sS -m 90 -L -A "$UA" -o "$f" "$url" 2>/dev/null && [ "$(wc -c < "$f")" -gt 4000 ] && [ "$(head -c 3 "$f" | od -An -tx1 | tr -d ' ')" = ffd8ff ]; then
          printf '%02d\t%s\t%s\t%s\n' "$n" "$group" "$(printf '%s' "$row" | cut -f2-)" "$query" >> "$DIR/credits.tsv"
          ok=1
        else
          rm -f "$f"
        fi
      fi
      [ "$ok" = 1 ] || sleep 4
    done
    if [ "$ok" = 1 ]; then echo "ok   $n $group: $query"; else echo "none $n $group: $query" >&2; fi
    sleep 3
  done < "$LIST"
  echo "fetched: $(ls "$DIR"/*.jpg 2>/dev/null | wc -l | tr -d ' ') of $(wc -l < "$LIST" | tr -d ' ') in $DIR"
  ;;
load)
  DIR=${2:?usage: test-gallery.sh load DIR URL TOKEN [blind]}; URL=${3:?}; TOKEN=${4:?}; BLIND=${5:-}
  n=0
  while IFS=$(printf '\t') read -r group query desc; do
    n=$((n + 1))
    f=$(pic "$n" "$group")
    [ -n "$f" ] || continue
    if [ "$BLIND" = blind ]; then
      code=$(curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -F "image=@$f" "$URL/gallery")
    else
      code=$(curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -F "image=@$f" --form-string "description=$desc" --form-string "group=$group" "$URL/gallery")
    fi
    echo "$code $(basename "$f")"
  done < "$LIST"
  ;;
score)
  DIR=${2:?usage: test-gallery.sh score DIR DATA}; DATA=${3:?}
  total=0; right=0; none=0
  i=0
  while IFS=$(printf '\t') read -r group query desc; do
    i=$((i + 1))
    f=$(pic "$i" "$group")
    [ -n "$f" ] || continue
    total=$((total + 1))
    # the pictures were loaded in the list's order, those that were fetched: the number is this one's place among them
    got=$(cat "$DATA/lib-$total.kind" 2>/dev/null || true)
    if [ -z "$got" ]; then none=$((none + 1)); printf 'none  %s: expected %s\n' "$(basename "$f")" "$group"
    elif [ "$got" = "$group" ]; then right=$((right + 1))
    else printf 'wrong %s: expected %s, got %s\n' "$(basename "$f")" "$group" "$got"; fi
  done < "$LIST"
  echo "score: $right of $total right ($none not sorted yet)"
  ;;
*) echo "usage: test-gallery.sh generate DIR URL TOKEN | collect DIR DATA | fetch DIR | load DIR URL TOKEN [blind] | score DIR DATA" >&2; exit 2 ;;
esac
