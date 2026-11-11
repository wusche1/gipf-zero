// gf1bridge <o|x> <searchdepth> <seed> [randomchoose=1]
// stdin:  "play <tokens>" applies decisions of either colour, "go" lets GF1 take the one decision it owns, "dump"
// tokens: a1-b2 (push)   r:b2-e5 (row to remove, only when several)   g:e5=y|n (one per GIPF piece of the removed row)
// stdout: "<tokens GF1 played>;<N|R|G|F> <decider> <37 cells b2..h5> <white> <black> <lostwhite> <lostblack> <winner> <ms>;<pending r:/g: questions>"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include "ai_gipf.h"
#include "board.h"
extern "C" void checkwindowevents (void) {}
extern "C" void srand (unsigned) throw () {} // GF1 reseeds from time() before every search
int interrupt_computer = 0;
static board * b;
static void * ai;
static double now () { struct timeval t; gettimeofday (&t, 0); return t.tv_sec * 1e3 + t.tv_usec / 1e3; }
static void die (const char * why, const char * what) { printf ("ERR %s %s\n", why, what); exit (1); }
static rem_row * row (int n) { return (rem_row *) llitembynr (b_row_extra (b), n); }
static rem_gipf * gipf (int n) { return (rem_gipf *) llitembynr (b_gipf_extra (b), n); }
static char decider () { return b_status (b) == S_REMOVEROW ? row_owner (row (1)) : b_status (b) == S_REMOVEGIPF ? b_gipf_owner (gipf (1)) : b_next_piece (b); }
static void dump (double ms)
{
    if (b_status (b) == S_NORMAL) b_game_finished (b);
    printf (";%c %c ", "NRGF"[b_status (b)], decider ());
    for (int c = 1; c < 8; c++) for (int r = 2; r <= b_colsize (c); r++) putchar (b->pieces[c][r]);
    printf (" %d %d %d %d %c %.1f;", b_white (b), b_black (b), b_white_lost (b), b_black_lost (b), b_winner (b), ms);
    rem_row * w; rem_gipf * g;
    if (b_status (b) == S_REMOVEROW) for (int n = 1; (w = row (n)); n++) printf ("r:%s-%s ", postostr (w->startpos), postostr (w->endpos));
    if (b_status (b) == S_REMOVEGIPF) for (int n = 1; (g = gipf (n)); n++) printf ("g:%s ", b_gipf_position (g));
    printf ("\n");
}
static void push (char * from, char * to)
{
    board * nb = b_move (b, from, to, b_next_piece (b));
    if (!nb) die ("illegal push", from);
    b = nb;
}
// answers: the g: tokens of this row (any order), or NULL to ask GF1
static int gipfs (char ** answers, int count)
{
    board * nb = b; rem_gipf * g; int n = 1;
    for (; (g = gipf (n)); n++)
    {
        char * p = b_gipf_position (g), r = 0;
        if (answers) { for (int i = 0; i < count; i++) if (!strncmp (answers[i], "g:", 2) && !strncmp (answers[i] + 2, p, 2)) r = answers[i][5]; }
        else { r = ai_gipf_gipf (b_copy (nb), ai, 0, p) == 'n' ? 'n' : 'y'; printf ("g:%s=%c ", p, r); }
        if (!r) die ("missing gipf answer", p);
        if (r == 'y') nb = b_remove_gipf (nb, g);
    }
    b = b_checkfour (nb);
    return n - 1;
}
int main (int argc, char ** argv)
{
    setvbuf (stdout, 0, _IOLBF, 0);
    char me = argv[1][0], line[256], * tok[64];
    b = b_new (T_STANDARD);
    ai = ai_gipf_new (me, T_STANDARD);
    ((ai_gipf *) ai)->searchdepth (atoi (argv[2]));
    ((ai_gipf *) ai)->random (argc > 4 ? atoi (argv[4]) : 1);
    srandom (atoi (argv[3]));
    while (fgets (line, sizeof line, stdin))
    {
        int n = 0; double t = now ();
        for (char * p = strtok (line, " \n"); p; p = strtok (0, " \n")) tok[n++] = p;
        if (n && !strcmp (tok[0], "play"))
            for (int i = 1; i < n; )
            {
                if (tok[i][0] == 'g' && tok[i][1] == ':') { if (b_status (b) != S_REMOVEGIPF) die ("no gipf pending", tok[i]); i += gipfs (tok + i, n - i); }
                else if (tok[i][0] == 'r' && tok[i][1] == ':')
                {
                    if (b_status (b) != S_REMOVEROW) die ("no row choice pending", tok[i]);
                    rem_row * w; int k = 1;
                    for (; (w = row (k)); k++)
                    {
                        char * s = postostr (w->startpos), * e = postostr (w->endpos);
                        if ((!strncmp (tok[i] + 2, s, 2) && !strncmp (tok[i] + 5, e, 2)) || (!strncmp (tok[i] + 2, e, 2) && !strncmp (tok[i] + 5, s, 2))) break;
                    }
                    if (!w) die ("no such row", tok[i]);
                    b = b_remove_row (b, k); i++;
                }
                else { if (b_status (b) != S_NORMAL) die ("capture pending", tok[i]); tok[i][2] = 0; push (tok[i], tok[i] + 3); i++; }
            }
        else if (n && !strcmp (tok[0], "go"))
        {
            if (decider () != me) die ("not my decision", "go");
            if (b_status (b) == S_NORMAL)
            {
                char type = 'n', from[3] = "  ", to[3] = "  ";
                ai_gipf_move (b_copy (b), ai, 0, &type, from, to);
                printf ("%s-%s ", from, to);
                push (from, to);
            }
            else
            {
                int ask = 1; // a GIPF question after the row choice may already belong to the next row
                if (b_status (b) == S_REMOVEROW)
                {
                    rem_row * w; int k = 1; board * tmp = b_copy (b);
                    for (; (w = row (k)); k++) if (ai_gipf_row (tmp, ai, 0, postostr (w->startpos), postostr (w->endpos)) == 'y') break;
                    if (!w) die ("GF1 chose no row", "go");
                    printf ("r:%s-%s ", postostr (w->startpos), postostr (w->endpos));
                    position * p; ask = 0;
                    for (int n = 1; (p = (position *) llitembynr (w->piecelist, n)); n++) ask |= b_ppiece (b, p) == 'O' || b_ppiece (b, p) == 'X';
                    b = b_remove_row (b, k);
                }
                if (ask && b_status (b) == S_REMOVEGIPF) gipfs (0, 0);
            }
        }
        dump (now () - t);
    }
    return 0;
}
