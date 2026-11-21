# rohaan-tactical — Mech Royale bot (Builders Buildathon)

Python bot for the mechfight/1 protocol. Plans paths on the map (A* around buildings), keeps inside the next
zone early, holds positions next to cover, faces and flank-strafes enemies, grabs laser > lightning > mortar,
retreats behind buildings to regenerate when low, and farms crates/drones for cubes.

Local check (from the bot kit folder):

    docker build -t my-bot .
    python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot

MIT License.
